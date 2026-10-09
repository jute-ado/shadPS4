// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <ranges>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "shader_recompiler/frontend/fetch_shader.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

namespace Shader::Gcn {
namespace {

u32 walker_calls{};

void PS4_SYSV_ABI CountingWalker(const u32* user_data, u32* flattened, PFN_SrtRangeRead, void*) {
    ++walker_calls;
    std::ranges::copy_n(user_data, NUM_USER_DATA_REGS, flattened);
}

template <typename T>
concept HasCompileFlattenedSnapshot = requires(T value) {
    value.compile_resource_snapshot;
};

struct GuestBytes {
    VAddr address{};
    void* data{};
    u64 size{};
    u32 calls{};
};

struct TessMemory {
    std::array<AmdGpu::Buffer, 2>* descriptors;
    TessellationDataConstantBuffer* constants_a;
    TessellationDataConstantBuffer* constants_b;
    u32 calls{};
};

bool PS4_SYSV_ABI ReadGuestBytes(void* context, VAddr address, void* output, u64 size) {
    auto& memory = *static_cast<GuestBytes*>(context);
    ++memory.calls;
    if (address < memory.address || size > memory.size || address - memory.address > memory.size - size) {
        return false;
    }
    std::memcpy(output, static_cast<const u8*>(memory.data) + (address - memory.address), size);
    return true;
}

bool PS4_SYSV_ABI ReadTessMemory(void* context, VAddr address, void* output, u64 size) {
    auto& memory = *static_cast<TessMemory*>(context);
    ++memory.calls;
    const auto copy_region = [&](VAddr base, const void* data, u64 region_size) {
        if (address < base || size > region_size || address - base > region_size - size) {
            return false;
        }
        std::memcpy(output, static_cast<const u8*>(data) + (address - base), size);
        return true;
    };
    return copy_region(reinterpret_cast<VAddr>(memory.descriptors->data()),
                       memory.descriptors->data(), sizeof(*memory.descriptors)) ||
           copy_region(reinterpret_cast<VAddr>(memory.constants_a), memory.constants_a,
                       sizeof(*memory.constants_a)) ||
           copy_region(reinterpret_cast<VAddr>(memory.constants_b), memory.constants_b,
                       sizeof(*memory.constants_b));
}

bool PS4_SYSV_ABI RefuseGuestBytes(void*, VAddr, void*, u64) {
    return false;
}

bool PS4_SYSV_ABI ReadZeroGuestBytes(void*, VAddr, void* output, u64 size) {
    std::memset(output, 0, size);
    return true;
}

template <typename T>
void StorePointer(std::array<u32, NUM_USER_DATA_REGS>& user_data, u32 index, T* pointer) {
    const auto address = reinterpret_cast<VAddr>(pointer);
    user_data[index] = static_cast<u32>(address);
    user_data[index + 1] = static_cast<u32>(address >> 32);
}

AmdGpu::Buffer MakeBuffer(VAddr address, u32 stride) {
    AmdGpu::Buffer buffer{};
    buffer.base_address = address;
    buffer.stride = stride;
    buffer.num_records = 32;
    return buffer;
}

TEST(CompileResourceSnapshot, ReplaysSelectedDescriptorGenerationWithoutCallingLiveWalker) {
    Info selected{};
    std::array<u32, NUM_USER_DATA_REGS> generation_a{};
    std::array<u32, NUM_USER_DATA_REGS> generation_b{};
    for (u32 index = 0; index < NUM_USER_DATA_REGS; ++index) {
        generation_a[index] = 0x1000U + index;
        generation_b[index] = 0x9000U + index;
    }
    selected.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS;
    selected.flattened_ud_buf.assign(generation_a.begin(), generation_a.end());
    const auto snapshot = CompileResourceSnapshot::Capture(selected);

    Info compiling{};
    compiling.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS;
    compiling.srt_info.walker_func = CountingWalker;
    compiling.user_data = generation_b;
    compiling.UseCompileResourceSnapshot(snapshot);
    walker_calls = 0;
    compiling.RefreshFlatBuf();

    EXPECT_EQ(walker_calls, 0U);
    EXPECT_TRUE(std::ranges::equal(compiling.flattened_ud_buf, generation_a));
}

TEST(CompileResourceSnapshot, ReplaysParsedFetchProgramWithoutReadingLiveGuestCode) {
    FetchShaderData selected{};
    selected.size = 24;
    selected.vertex_offset_sgpr = 6;
    selected.instance_offset_sgpr = 8;
    selected.attributes.push_back(VertexAttribute{.semantic = 3, .dest_vgpr = 5});

    Info compiling{};
    compiling.has_fetch_shader = true;
    compiling.UseCompileFetchShaderSnapshot(selected);

    ASSERT_TRUE(ParseFetchShader(compiling).has_value());
    EXPECT_EQ(*ParseFetchShader(compiling), selected);
}

TEST(CompileResourceSnapshot, ReplaysIndirectUserDataAfterRootsAndPointeesMutate) {
    std::array<AmdGpu::Buffer, 2> descriptors{
        MakeBuffer(0x11110000, 16),
        MakeBuffer(0x99990000, 64),
    };
    std::array<u32, NUM_USER_DATA_REGS> roots{};
    StorePointer(roots, 0, &descriptors[0]);

    GuestBytes memory{
        .address = reinterpret_cast<VAddr>(descriptors.data()),
        .data = descriptors.data(),
        .size = sizeof(descriptors),
    };
    CompileResourceCapture capture{{ReadGuestBytes, &memory}};
    Info selected{};
    selected.user_data = roots;
    selected.flattened_ud_buf.assign(roots.begin(), roots.end());
    selected.UseCompileCaptureReader(capture.Reader());
    const VertexAttribute attribute{.sgpr_base = 0, .dword_offset = 0};
    ASSERT_EQ(attribute.GetSharp(selected), descriptors[0]);
    auto snapshot = capture.Finish(selected);
    const u32 capture_calls = memory.calls;

    descriptors[0] = descriptors[1];
    StorePointer(roots, 0, &descriptors[1]);
    Info replay{};
    replay.user_data = roots;
    replay.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS;
    replay.UseCompileResourceSnapshot(snapshot);

    EXPECT_EQ(attribute.GetSharp(replay), MakeBuffer(0x11110000, 16));
    EXPECT_EQ(memory.calls, capture_calls);
}

TEST(CompileResourceSnapshot, FailedNestedSrtReadInvalidatesTheCapturedGeneration) {
    constexpr VAddr DescriptorTable = 0x285208E10ULL;
    std::array<u32, NUM_USER_DATA_REGS> roots{};
    roots[0] = static_cast<u32>(DescriptorTable);
    roots[1] = static_cast<u32>(DescriptorTable >> 32);

    Info selected{};
    selected.user_data = roots;
    selected.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS + 4;
    selected.srt_info.range_reservations.push_back({
        .pointer_lo_flat_index = 0,
        .source_offset_dw = 0,
        .destination_flat_offset_dw = NUM_USER_DATA_REGS,
        .span_dw = 4,
    });
    CompileResourceCapture capture{{RefuseGuestBytes, nullptr}};

    selected.RefreshFlatBuf(capture.Reader());
    const auto snapshot = capture.Finish(selected);

    ASSERT_EQ(snapshot.flattened_user_data.size(), NUM_USER_DATA_REGS + 4);
    EXPECT_TRUE(std::ranges::all_of(snapshot.flattened_user_data | std::views::drop(NUM_USER_DATA_REGS),
                                   [](u32 word) { return word == 0; }));
    EXPECT_FALSE(snapshot.valid);
}

TEST(CompileResourceSnapshot, SuccessfulNestedSrtReadOfRealZerosRemainsValid) {
    constexpr VAddr DescriptorTable = 0x285208E10ULL;
    std::array<u32, NUM_USER_DATA_REGS> roots{};
    roots[0] = static_cast<u32>(DescriptorTable);
    roots[1] = static_cast<u32>(DescriptorTable >> 32);

    Info selected{};
    selected.user_data = roots;
    selected.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS + 4;
    selected.srt_info.range_reservations.push_back({
        .pointer_lo_flat_index = 0,
        .source_offset_dw = 0,
        .destination_flat_offset_dw = NUM_USER_DATA_REGS,
        .span_dw = 4,
    });
    CompileResourceCapture capture{{ReadZeroGuestBytes, nullptr}};

    selected.RefreshFlatBuf(capture.Reader());
    const auto snapshot = capture.Finish(selected);

    ASSERT_EQ(snapshot.flattened_user_data.size(), NUM_USER_DATA_REGS + 4);
    EXPECT_TRUE(std::ranges::all_of(snapshot.flattened_user_data | std::views::drop(NUM_USER_DATA_REGS),
                                   [](u32 word) { return word == 0; }));
    EXPECT_TRUE(snapshot.valid);
}

TEST(CompileResourceSnapshot, EpochInvalidationAfterTheLastReadRejectsPublication) {
    Info selected{};
    selected.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS;
    selected.flattened_ud_buf.resize(NUM_USER_DATA_REGS);
    CompileResourceCapture capture{{ReadZeroGuestBytes, nullptr}};

    const auto snapshot = capture.Finish(selected, false);

    EXPECT_FALSE(snapshot.valid);
}

TEST(CompileResourceSnapshot, OwnsReplayBytesAfterTheSourceSnapshotChanges) {
    std::array<AmdGpu::Buffer, 2> descriptors{
        MakeBuffer(0x11110000, 16),
        MakeBuffer(0x99990000, 64),
    };
    std::array<u32, NUM_USER_DATA_REGS> generation_a{};
    std::array<u32, NUM_USER_DATA_REGS> generation_b{};
    StorePointer(generation_a, 0, &descriptors[0]);
    StorePointer(generation_b, 0, &descriptors[1]);

    CompileResourceSnapshot snapshot{
        .flattened_user_data = {generation_a.begin(), generation_a.end()},
        .valid = true,
    };
    snapshot.RecordGuestMemory(reinterpret_cast<VAddr>(&descriptors[0]), &descriptors[0],
                               sizeof(descriptors[0]));

    Info replay{};
    replay.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS;
    replay.UseCompileResourceSnapshot(snapshot);
    Info copied_replay = replay;

    snapshot.flattened_user_data.assign(generation_b.begin(), generation_b.end());
    snapshot.guest_reads.clear();
    replay.ClearCompileResourceAccess();

    const VertexAttribute attribute{.sgpr_base = 0, .dword_offset = 0};
    EXPECT_EQ(attribute.GetSharp(copied_replay), MakeBuffer(0x11110000, 16));
}

TEST(CompileResourceSnapshot, AsyncPipelineJobRebindKeepsOwnedReplayState) {
    AmdGpu::Buffer descriptor = MakeBuffer(0x12340000, 48);
    std::array<u32, NUM_USER_DATA_REGS> roots{};
    StorePointer(roots, 0, &descriptor);
    CompileResourceSnapshot snapshot{
        .flattened_user_data = {roots.begin(), roots.end()},
        .valid = true,
    };
    snapshot.RecordGuestMemory(reinterpret_cast<VAddr>(&descriptor), &descriptor,
                               sizeof(descriptor));

    Info source{};
    source.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS;
    source.UseCompileResourceSnapshot(snapshot);
    FetchShaderData fetch{};
    source.UseCompileFetchShaderSnapshot(fetch);

    Vulkan::AsyncGraphicsPipelineJob job{};
    job.retained_infos[0] = &source;
    job.info_copies[0] = source;
    job.user_data_sizes[0] = NUM_USER_DATA_REGS;
    job.user_data_copies[0] = roots;
    job.RebindCompileInfo();

    source.ClearCompileResourceAccess();
    snapshot.guest_reads.clear();
    ASSERT_EQ(job.compile_infos[0], &job.info_copies[0]);
    EXPECT_EQ(job.info_copies[0].user_data.data(), job.user_data_copies[0].data());
    EXPECT_EQ(job.info_copies[0].compile_fetch_shader_snapshot, nullptr);
    const VertexAttribute attribute{.sgpr_base = 0, .dword_offset = 0};
    EXPECT_EQ(attribute.GetSharp(*job.compile_infos[0]), MakeBuffer(0x12340000, 48));
}

TEST(CompileResourceSnapshot, MissingReplayRangeNeverFallsBackToLiveGuestMemory) {
    AmdGpu::Buffer live = MakeBuffer(0x77770000, 32);
    std::array<u32, NUM_USER_DATA_REGS> roots{};
    StorePointer(roots, 0, &live);

    Info selected{};
    selected.user_data = roots;
    selected.flattened_ud_buf.assign(roots.begin(), roots.end());
    const auto snapshot = CompileResourceSnapshot::Capture(selected);

    Info replay{};
    replay.user_data = roots;
    replay.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS;
    replay.UseCompileResourceSnapshot(snapshot);
    const VertexAttribute attribute{.sgpr_base = 0, .dword_offset = 0};

    EXPECT_FALSE(attribute.GetSharp(replay));
}

TEST(CompileResourceSnapshot, ReplaysTessellationDescriptorAndPayloadFromOneGeneration) {
    TessellationDataConstantBuffer constants_a{};
    constants_a.ls_stride = 16;
    constants_a.hs_cp_stride = 32;
    TessellationDataConstantBuffer constants_b{};
    constants_b.ls_stride = 80;
    constants_b.hs_cp_stride = 96;
    std::array<AmdGpu::Buffer, 2> descriptors{
        MakeBuffer(reinterpret_cast<VAddr>(&constants_a), sizeof(constants_a)),
        MakeBuffer(reinterpret_cast<VAddr>(&constants_b), sizeof(constants_b)),
    };
    std::array<u32, NUM_USER_DATA_REGS> roots{};
    StorePointer(roots, 0, &descriptors[0]);

    TessMemory memory{&descriptors, &constants_a, &constants_b};
    CompileResourceCapture capture{{ReadTessMemory, &memory}};
    Info selected{};
    selected.user_data = roots;
    selected.flattened_ud_buf.assign(roots.begin(), roots.end());
    selected.tess_consts_ptr_base = static_cast<IR::ScalarReg>(0);
    selected.tess_consts_dword_offset = 0;
    selected.UseCompileCaptureReader(capture.Reader());
    TessellationDataConstantBuffer captured{};
    selected.ReadTessConstantBuffer(captured);
    ASSERT_EQ(captured.ls_stride, constants_a.ls_stride);
    auto snapshot = capture.Finish(selected);
    const u32 capture_calls = memory.calls;

    descriptors[0] = descriptors[1];
    constants_a = constants_b;
    StorePointer(roots, 0, &descriptors[1]);
    Info replay{};
    replay.user_data = roots;
    replay.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS;
    replay.tess_consts_ptr_base = static_cast<IR::ScalarReg>(0);
    replay.tess_consts_dword_offset = 0;
    replay.UseCompileResourceSnapshot(snapshot);
    TessellationDataConstantBuffer replayed{};
    replay.ReadTessConstantBuffer(replayed);

    EXPECT_EQ(replayed.ls_stride, 16U);
    EXPECT_EQ(replayed.hs_cp_stride, 32U);
    EXPECT_EQ(memory.calls, capture_calls);
}

TEST(CompileResourceSnapshot, CachedPermutationCompileConsumesSelectedGeneration) {
    std::ifstream stream{SHADPS4_PIPELINE_CACHE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const std::string source{std::istreambuf_iterator<char>{stream},
                             std::istreambuf_iterator<char>{}};
    const auto get_program = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto capture = source.find("CompileResourceCapture compile_capture", get_program);
    const auto capture_reader = source.find("info.UseCompileCaptureReader", get_program);
    const auto finish = source.find("compile_capture.Finish(info, epoch_valid)", get_program);
    const auto async_translate =
        source.find("TranslateModule(*job->pools, job->info", finish);
    const auto sync_translate = source.find("TranslateModule(pools, new_info", finish);
    const auto async_submit = source.find("SubmitAsyncShaderModule", async_translate);
    const auto sync_compile = source.find("module = CompileModule(new_info", sync_translate);
    ASSERT_NE(get_program, std::string::npos);
    ASSERT_NE(capture, std::string::npos);
    ASSERT_NE(capture_reader, std::string::npos);
    ASSERT_NE(finish, std::string::npos);
    ASSERT_NE(async_translate, std::string::npos);
    ASSERT_NE(sync_translate, std::string::npos);
    ASSERT_NE(async_submit, std::string::npos);
    ASSERT_NE(sync_compile, std::string::npos);
    EXPECT_LT(capture, async_translate);
    EXPECT_LT(capture_reader, finish);
    EXPECT_LT(finish, async_translate);
    EXPECT_LT(async_translate, async_submit);
    EXPECT_LT(finish, sync_translate);
    EXPECT_LT(sync_translate, sync_compile);
    EXPECT_NE(source.substr(async_translate, 400).find("&compile_resources"), std::string::npos);
    EXPECT_NE(source.substr(sync_translate, 400).find("&compile_resources"), std::string::npos);
}

TEST(CompileResourceSnapshot, FailedCaptureCannotSelectOrPublishAPermutation) {
    std::ifstream stream{SHADPS4_PIPELINE_CACHE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const std::string source{std::istreambuf_iterator<char>{stream},
                             std::istreambuf_iterator<char>{}};
    const auto get_program = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto replace_shader =
        source.find("std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(", get_program);
    ASSERT_NE(get_program, std::string::npos);
    ASSERT_NE(replace_shader, std::string::npos);
    const auto body = source.substr(get_program, replace_shader - get_program);

    const auto first_finish = body.find("compile_capture.Finish(program->info, epoch_valid)");
    const auto first_invalid_capture = body.find("if (!compile_resources.valid", first_finish);
    const auto first_publish = body.find("RegisterShaderMeta(program->info", first_finish);
    const auto repeated_finish = body.find("compile_capture.Finish(info, epoch_valid)");
    const auto invalid_capture = body.find("if (!compile_resources.valid", repeated_finish);
    const auto lookup = body.find("program->FindPermut(spec)", repeated_finish);
    const auto publish = body.find("RegisterShaderMeta(new_info", repeated_finish);

    ASSERT_NE(first_finish, std::string::npos);
    ASSERT_NE(first_invalid_capture, std::string::npos);
    ASSERT_NE(first_publish, std::string::npos);
    ASSERT_NE(repeated_finish, std::string::npos);
    ASSERT_NE(invalid_capture, std::string::npos);
    ASSERT_NE(lookup, std::string::npos);
    ASSERT_NE(publish, std::string::npos);
    EXPECT_LT(first_finish, first_invalid_capture);
    EXPECT_LT(first_invalid_capture, first_publish);
    EXPECT_LT(first_publish, repeated_finish);
    EXPECT_LT(repeated_finish, invalid_capture);
    EXPECT_LT(invalid_capture, lookup);
    EXPECT_LT(invalid_capture, publish);
}

TEST(CompileResourceSnapshot, RefusedReadStopsBeforeSpirvOrShaderBinaryPublication) {
    std::ifstream stream{SHADPS4_PIPELINE_CACHE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const std::string source{std::istreambuf_iterator<char>{stream},
                             std::istreambuf_iterator<char>{}};
    const auto translate_module =
        source.find("std::optional<Shader::IR::Program> PipelineCache::TranslateModule(");
    const auto translate = source.find("Shader::TranslateProgram(", translate_module);
    const auto invalid_reader = source.find("if (!compile_reader.Valid()", translate);
    const auto compile_module =
        source.find("vk::ShaderModule PipelineCache::CompileModule(", invalid_reader);
    const auto emit = source.find("Shader::Backend::SPIRV::EmitSPIRV", compile_module);
    const auto publish = source.find("RegisterShaderBinary", compile_module);

    ASSERT_NE(translate_module, std::string::npos);
    ASSERT_NE(compile_module, std::string::npos);
    ASSERT_NE(translate, std::string::npos);
    ASSERT_NE(invalid_reader, std::string::npos);
    ASSERT_NE(emit, std::string::npos);
    ASSERT_NE(publish, std::string::npos);
    EXPECT_LT(translate, invalid_reader);
    EXPECT_LT(invalid_reader, compile_module);
    EXPECT_LT(invalid_reader, emit);
    EXPECT_LT(invalid_reader, publish);
}

TEST(CompileResourceSnapshot, FailedSelectionRetriesFromFreshBindingStateBeforeGivingUp) {
    std::ifstream stream{SHADPS4_PIPELINE_CACHE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const std::string source{std::istreambuf_iterator<char>{stream},
                             std::istreambuf_iterator<char>{}};
    const auto get_program = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto get_attempt = source.find("PipelineCache::Result PipelineCache::GetProgramAttempt(");
    ASSERT_NE(get_program, std::string::npos);
    ASSERT_NE(get_attempt, std::string::npos);
    ASSERT_LT(get_program, get_attempt);
    const auto body = source.substr(get_program, get_attempt - get_program);

    const auto budget = body.find("MaxCaptureAttempts = 2");
    const auto initial = body.find("const auto initial_binding = binding");
    const auto loop = body.find("for (u32 attempt = 0; attempt < MaxCaptureAttempts; ++attempt)");
    const auto restore = body.find("binding = initial_binding", loop);
    const auto select = body.find(
        "GetProgramAttempt(stage, l_stage, params, binding, srt_read_session, attempt_pending)",
        restore);
    const auto accept = body.find("std::get<0>(result) != nullptr", select);
    const auto pending = body.find("if (attempt_pending)", accept);
    const auto pending_break = body.find("break", pending);
    const auto final_restore = body.find("binding = initial_binding", pending_break);

    ASSERT_NE(budget, std::string::npos);
    ASSERT_NE(initial, std::string::npos);
    ASSERT_NE(loop, std::string::npos);
    ASSERT_NE(restore, std::string::npos);
    ASSERT_NE(select, std::string::npos);
    ASSERT_NE(accept, std::string::npos);
    ASSERT_NE(pending, std::string::npos);
    ASSERT_NE(pending_break, std::string::npos);
    ASSERT_NE(final_restore, std::string::npos);
    EXPECT_LT(initial, loop);
    EXPECT_LT(restore, select);
    EXPECT_LT(select, accept);
    EXPECT_LT(accept, pending);
    EXPECT_LT(pending, pending_break);
    EXPECT_LT(accept, final_restore);
}

TEST(CompileResourceSnapshot, FailedFragmentSelectionCannotPolluteLaterStagePermutations) {
    std::ifstream stream{SHADPS4_PIPELINE_CACHE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const std::string source{std::istreambuf_iterator<char>{stream},
                             std::istreambuf_iterator<char>{}};
    const auto refresh = source.find("bool PipelineCache::RefreshGraphicsStages()");
    const auto fragment = source.find("bind_stage(Stage::Fragment, LogicalStage::Fragment)", refresh);
    const auto fragment_abort = source.find("if (stage_selection_failed)", fragment);
    const auto stage_switch = source.find("switch (regs.stage_enable.raw)", fragment);

    ASSERT_NE(refresh, std::string::npos);
    ASSERT_NE(fragment, std::string::npos);
    ASSERT_NE(fragment_abort, std::string::npos);
    ASSERT_NE(stage_switch, std::string::npos);
    EXPECT_LT(fragment, fragment_abort);
    EXPECT_LT(fragment_abort, stage_switch);
}

TEST(CompileResourceSnapshot, CompileOnlyStateIsExcludedFromPersistentShaderCacheAbi) {
    static_assert(std::is_base_of_v<InfoPersistent, Info>);
    static_assert(!HasCompileFlattenedSnapshot<InfoPersistent>);
    SUCCEED();
}

TEST(ShaderPermutationInfo, RetainsTheLayoutThatProducedEachModuleAcrossListGrowth) {
    Vulkan::Program program{};
    const Shader::Info* first_info{};

    for (u64 index = 0; index <= Vulkan::Program::MaxPermutations; ++index) {
        Info compiled_info{};
        compiled_info.pgm_hash = 0x1000 + index;
        StageSpecialization spec{};
        program.AddPermut(vk::ShaderModule{}, std::move(spec), std::move(compiled_info));
        const auto& retained = program.GetPermutInfo(index);
        EXPECT_EQ(retained.pgm_hash, 0x1000 + index);
        EXPECT_EQ(program.modules[index].spec.info, &retained);
        if (index == 0) {
            first_info = &retained;
        }
    }

    EXPECT_EQ(first_info, &program.GetPermutInfo(0));
    EXPECT_EQ(first_info->pgm_hash, 0x1000U);
}

TEST(ShaderPermutationInfo, BoundAndUnboundSelectionIsIndependentOfInsertionOrder) {
    constexpr u64 BoundIdentity = 0xB001U;
    constexpr u64 UnboundIdentity = 0xA001U;

    struct SelectionResult {
        size_t module_count{};
        u64 bound_identity{};
        u64 unbound_identity{};
        bool modules_are_distinct{};
    };

    const auto select_in_order = [&](bool bound_first) {
        Info source_info{};
        source_info.buffers.push_back(BufferResource{
            .sharp_idx = 0,
            .used_types = IR::Type::U32,
            .inline_cbuf = {},
            .buffer_type = BufferType::Guest,
        });
        const auto bound_words = std::bit_cast<std::array<u32, 4>>(MakeBuffer(1, 4));
        source_info.flattened_ud_buf.assign(bound_words.begin(), bound_words.end());

        const RuntimeInfo runtime_info{};
        const Profile profile{};
        const Backend::Bindings bindings{};
        const StageSpecialization bound{source_info, runtime_info, profile, bindings};

        std::ranges::fill(source_info.flattened_ud_buf, 0U);
        const StageSpecialization unbound{source_info, runtime_info, profile, bindings};

        Vulkan::Program program{};
        const auto select = [&](StageSpecialization spec, u64 identity) {
            auto it = program.FindPermut(spec);
            if (it == program.modules.end()) {
                Info compiled_info{};
                compiled_info.pgm_hash = identity;
                program.AddPermut(vk::ShaderModule{}, std::move(spec), std::move(compiled_info));
                it = std::prev(program.modules.end());
            }
            return it->info->pgm_hash;
        };

        if (bound_first) {
            select(bound, BoundIdentity);
            select(unbound, UnboundIdentity);
        } else {
            select(unbound, UnboundIdentity);
            select(bound, BoundIdentity);
        }

        const auto bound_it = program.FindPermut(bound);
        const auto unbound_it = program.FindPermut(unbound);
        return SelectionResult{
            .module_count = program.modules.size(),
            .bound_identity =
                bound_it != program.modules.end() ? bound_it->info->pgm_hash : 0U,
            .unbound_identity =
                unbound_it != program.modules.end() ? unbound_it->info->pgm_hash : 0U,
            .modules_are_distinct = bound_it != unbound_it,
        };
    };

    const auto full_then_sparse = select_in_order(true);
    const auto sparse_then_full = select_in_order(false);

    EXPECT_EQ(full_then_sparse.module_count, 2U);
    EXPECT_EQ(sparse_then_full.module_count, 2U);
    EXPECT_TRUE(full_then_sparse.modules_are_distinct);
    EXPECT_TRUE(sparse_then_full.modules_are_distinct);
    EXPECT_EQ(full_then_sparse.bound_identity, BoundIdentity);
    EXPECT_EQ(full_then_sparse.unbound_identity, UnboundIdentity);
    EXPECT_EQ(sparse_then_full.bound_identity, BoundIdentity);
    EXPECT_EQ(sparse_then_full.unbound_identity, UnboundIdentity);
    EXPECT_EQ(full_then_sparse.bound_identity, sparse_then_full.bound_identity);
    EXPECT_EQ(full_then_sparse.unbound_identity, sparse_then_full.unbound_identity);
}

TEST(ShaderPermutationInfo, UnboundResourcesStillSpecializeDescriptorBindingStart) {
    Info info{};
    StageSpecialization first{};
    StageSpecialization second{};
    first.info = &info;
    second.info = &info;
    first.user_data_mask = info.ud_mask.mask;
    second.user_data_mask = info.ud_mask.mask;
    first.images.resize(1);
    second.images.resize(1);
    first.start.unified = 3;
    second.start.unified = 7;

    EXPECT_FALSE(first == second);
    EXPECT_FALSE(second == first);
}

TEST(ShaderPermutationInfo, SamplerForceStateIsComparedWithoutBoundBuffersOrImages) {
    Info info{};
    StageSpecialization first{};
    StageSpecialization second{};
    first.info = &info;
    second.info = &info;
    first.samplers.push_back({.force_unnormalized = false, .force_degamma = false});
    second.samplers.push_back({.force_unnormalized = true, .force_degamma = false});

    EXPECT_FALSE(first == second);
    EXPECT_FALSE(second == first);
}

TEST(ShaderPermutationInfo, UserDataAccessSpecializesUserDataBindingStart) {
    Info info{};
    info.ud_mask.Set(IR::ScalarReg::S0);
    StageSpecialization first{};
    StageSpecialization second{};
    first.info = &info;
    second.info = &info;
    first.user_data_mask = info.ud_mask.mask;
    second.user_data_mask = info.ud_mask.mask;
    first.start.user_data = 2;
    second.start.user_data = 5;

    EXPECT_FALSE(first == second);
    EXPECT_FALSE(second == first);
}

TEST(ShaderPermutationInfo, RetainedUserDataLayoutDoesNotMutateWhenInfoPointerIsRebound) {
    Info original_info{};
    original_info.ud_mask.Set(IR::ScalarReg::S0);
    Info rebound_info{};
    rebound_info.ud_mask.Set(IR::ScalarReg::S1);

    const RuntimeInfo runtime_info{};
    const Profile profile{};
    const Backend::Bindings bindings{};
    StageSpecialization retained{original_info, runtime_info, profile, bindings};
    const StageSpecialization original_identity{original_info, runtime_info, profile, bindings};
    const StageSpecialization rebound_identity{rebound_info, runtime_info, profile, bindings};

    retained.info = &rebound_info;

    EXPECT_TRUE(retained == original_identity);
    EXPECT_TRUE(original_identity == retained);
    EXPECT_FALSE(retained == rebound_identity);
    EXPECT_FALSE(rebound_identity == retained);
}

TEST(ShaderPermutationInfo, ResourceListCardinalityIsPartOfPermutationLayout) {
    Info info{};
    StageSpecialization first{};
    StageSpecialization second{};
    first.info = &info;
    second.info = &info;
    first.buffers.resize(1);

    EXPECT_FALSE(first == second);
    EXPECT_FALSE(second == first);
}

TEST(ShaderPermutationInfo, UnboundDynamicImageArrayCountIsPartOfPermutationLayout) {
    Info first_info{};
    Info second_info{};
    first_info.stage = Stage::Compute;
    second_info.stage = Stage::Compute;
    first_info.l_stage = LogicalStage::Compute;
    second_info.l_stage = LogicalStage::Compute;
    first_info.flattened_ud_buf.resize(8, 0U);
    second_info.flattened_ud_buf.resize(8, 0U);

    ImageResource first_image{};
    first_image.dynamic_descriptor_count = 2;
    first_image.dynamic_descriptor_stride_dw = 8;
    ImageResource second_image = first_image;
    second_image.dynamic_descriptor_count = 4;
    first_info.images.push_back(first_image);
    second_info.images.push_back(second_image);

    const RuntimeInfo runtime_info{};
    const Profile profile{};
    const Backend::Bindings bindings{};
    const StageSpecialization first{first_info, runtime_info, profile, bindings};
    const StageSpecialization second{second_info, runtime_info, profile, bindings};

    ASSERT_TRUE(first.bitset.none());
    ASSERT_TRUE(second.bitset.none());
    EXPECT_FALSE(first == second);
    EXPECT_FALSE(second == first);
}

TEST(ShaderPermutationInfo, UnboundSampledAndStorageImagesHaveDistinctLayouts) {
    Info sampled_info{};
    Info storage_info{};
    sampled_info.stage = Stage::Compute;
    storage_info.stage = Stage::Compute;
    sampled_info.l_stage = LogicalStage::Compute;
    storage_info.l_stage = LogicalStage::Compute;
    sampled_info.flattened_ud_buf.resize(8, 0U);
    storage_info.flattened_ud_buf.resize(8, 0U);

    ImageResource sampled_image{};
    ImageResource storage_image = sampled_image;
    storage_image.is_written = true;
    sampled_info.images.push_back(sampled_image);
    storage_info.images.push_back(storage_image);

    const RuntimeInfo runtime_info{};
    const Profile profile{};
    const Backend::Bindings bindings{};
    const StageSpecialization sampled{sampled_info, runtime_info, profile, bindings};
    const StageSpecialization storage{storage_info, runtime_info, profile, bindings};

    ASSERT_TRUE(sampled.bitset.none());
    ASSERT_TRUE(storage.bitset.none());
    EXPECT_FALSE(sampled == storage);
    EXPECT_FALSE(storage == sampled);
}

TEST(ShaderPermutationInfo, PreloadReusesTheExactPermutationWithoutReplacingRetainedInfo) {
    Vulkan::Program program{};
    Info compiled_info{};
    compiled_info.pgm_hash = 0xB001U;
    StageSpecialization stored_spec{};
    stored_spec.info = &compiled_info;
    program.AddPermut(vk::ShaderModule{}, std::move(stored_spec), std::move(compiled_info));

    const auto* retained_info = &program.GetPermutInfo(0);
    const auto* retained_module = &program.modules[0];
    StageSpecialization same_spec = program.modules[0].spec;

    const auto decision = program.InspectPreloadPermutation(same_spec, 0);

    EXPECT_EQ(decision.action, Vulkan::Program::PreloadAction::Reuse);
    EXPECT_EQ(decision.module, retained_module);
    Info serialized_layout{};
    serialized_layout.pgm_hash = 0xD001U;
    EXPECT_FALSE(program.InsertPreloadPermutation(vk::ShaderModule{}, std::move(same_spec),
                                                  std::move(serialized_layout), 0));
    EXPECT_EQ(&program.GetPermutInfo(0), retained_info);
    EXPECT_EQ(program.modules[0].spec.info, retained_info);
    EXPECT_EQ(retained_info->pgm_hash, 0xB001U);
}

TEST(ShaderPermutationInfo, PreloadRefusesToReplaceAnOccupiedPermutationIndex) {
    Vulkan::Program program{};
    Info retained_layout{};
    retained_layout.pgm_hash = 0xB001U;
    StageSpecialization retained_spec{};
    retained_spec.info = &retained_layout;
    program.AddPermut(vk::ShaderModule{}, std::move(retained_spec), std::move(retained_layout));
    const auto* retained_info = &program.GetPermutInfo(0);

    Info conflicting_layout{};
    conflicting_layout.pgm_hash = 0xC001U;
    StageSpecialization conflicting_spec{};
    conflicting_spec.info = &conflicting_layout;
    conflicting_spec.runtime_info.fs_info.num_inputs = 1;

    const auto decision = program.InspectPreloadPermutation(conflicting_spec, 0);
    EXPECT_EQ(decision.action, Vulkan::Program::PreloadAction::Conflict);
    EXPECT_EQ(decision.module, nullptr);
    EXPECT_FALSE(program.InsertPreloadPermutation(vk::ShaderModule{}, std::move(conflicting_spec),
                                                  std::move(conflicting_layout), 0));
    EXPECT_EQ(&program.GetPermutInfo(0), retained_info);
    EXPECT_EQ(retained_info->pgm_hash, 0xB001U);
}

TEST(ShaderPermutationInfo, PreloadRefusesAMatchingPermutationStoredAtAnotherIndex) {
    Vulkan::Program program{};
    Info retained_layout{};
    retained_layout.pgm_hash = 0xB001U;
    StageSpecialization retained_spec{};
    retained_spec.info = &retained_layout;
    program.AddPermut(vk::ShaderModule{}, std::move(retained_spec), std::move(retained_layout));
    const auto* retained_info = &program.GetPermutInfo(0);
    StageSpecialization same_spec = program.modules[0].spec;

    const auto decision = program.InspectPreloadPermutation(same_spec, 2);

    EXPECT_EQ(decision.action, Vulkan::Program::PreloadAction::Conflict);
    EXPECT_EQ(decision.module, nullptr);
    Info serialized_layout{};
    serialized_layout.pgm_hash = 0xD001U;
    EXPECT_FALSE(program.InsertPreloadPermutation(vk::ShaderModule{}, std::move(same_spec),
                                                  std::move(serialized_layout), 2));
    EXPECT_EQ(program.modules.size(), 1U);
    EXPECT_EQ(&program.GetPermutInfo(0), retained_info);
}

TEST(ShaderPermutationInfo, CachedSelectionUsesTheSelectedPermutationLayout) {
    std::ifstream stream{SHADPS4_PIPELINE_CACHE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const std::string source{std::istreambuf_iterator<char>{stream},
                             std::istreambuf_iterator<char>{}};
    const auto get_program_attempt =
        source.find("PipelineCache::Result PipelineCache::GetProgramAttempt(");
    const auto replace_shader =
        source.find("std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(",
                    get_program_attempt);
    ASSERT_NE(get_program_attempt, std::string::npos);
    ASSERT_NE(replace_shader, std::string::npos);
    const auto body =
        source.substr(get_program_attempt, replace_shader - get_program_attempt);
    EXPECT_NE(body.find("GetPermutInfo(perm_idx)"), std::string::npos);
    EXPECT_NE(body.find("const bool new_permutation"), std::string::npos);
    EXPECT_NE(body.find("if (!new_permutation)"), std::string::npos);
    EXPECT_NE(body.find("selected_info.AddBindings(binding)"), std::string::npos);
    EXPECT_NE(body.find("&selected_info"), std::string::npos);
}

} // namespace
} // namespace Shader::Gcn
