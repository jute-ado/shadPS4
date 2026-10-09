// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <iterator>
#include <ranges>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "shader_recompiler/frontend/fetch_shader.h"

namespace Shader::Gcn {
namespace {

u32 walker_calls{};

void PS4_SYSV_ABI CountingWalker(const u32* user_data, u32* flattened) {
    ++walker_calls;
std::ranges::copy_n(user_data, NUM_USER_DATA_REGS, flattened);
}

AmdGpu::Buffer MakeVertexBuffer(u64 address, u32 stride, u32 records, u32 format) {
    AmdGpu::Buffer buffer{};
    buffer.base_address = address;
    buffer.stride = stride;
    buffer.num_records = records;
    buffer.data_format = format;
    buffer.num_format = 1;
    buffer.dst_sel_x = 0;
    buffer.dst_sel_y = 1;
    buffer.dst_sel_z = 2;
    buffer.dst_sel_w = 3;
    return buffer;
}

std::string ReadSource(const char* path) {
    std::ifstream stream{path, std::ios::binary};
    EXPECT_TRUE(stream.is_open());
    return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

template <typename T>
concept HasCompileFlattenedSnapshot = requires(T value) {
    value.compile_flattened_ud_snapshot;
};

template <typename T>
concept HasCompileFetchSnapshot = requires(T value) {
    value.compile_fetch_shader_snapshot;
};

TEST(CompileResourceSnapshot, ReplaysOneCapturedDescriptorGenerationWithoutCallingTheLiveWalker) {
    Info selected{};
    std::array<u32, NUM_USER_DATA_REGS> generation_a{};
    std::array<u32, NUM_USER_DATA_REGS> generation_b{};
    for (u32 index = 0; index < NUM_USER_DATA_REGS; ++index) {
        generation_a[index] = 0x1000u + index;
        generation_b[index] = 0x9000u + index;
    }
    selected.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS;
    selected.flattened_ud_buf.assign(generation_a.begin(), generation_a.end());

    const auto snapshot = CompileResourceSnapshot::Capture(selected);
    ASSERT_TRUE(snapshot.valid);

    Info compiling{};
    compiling.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS;
    compiling.srt_info.walker_func = CountingWalker;
    compiling.user_data = generation_b;
    compiling.UseCompileResourceSnapshot(snapshot);
    walker_calls = 0;

    compiling.RefreshFlatBuf();

    EXPECT_EQ(walker_calls, 0u);
    EXPECT_TRUE(std::ranges::equal(compiling.flattened_ud_buf, generation_a));
    EXPECT_FALSE(std::ranges::equal(compiling.flattened_ud_buf, generation_b));
}

TEST(CompileResourceSnapshot, RejectsMalformedDescriptorImagesBeforeReplay) {
    Info selected{};
    selected.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS;
    selected.flattened_ud_buf.assign(NUM_USER_DATA_REGS - 1, 0x55aa55aau);

    const auto malformed = CompileResourceSnapshot::Capture(selected);
    EXPECT_FALSE(malformed.valid);

    Info compiling{};
    compiling.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS;
    compiling.UseCompileResourceSnapshot(malformed);
    EXPECT_FALSE(compiling.ReplayCompileResourceSnapshot());
    EXPECT_TRUE(compiling.flattened_ud_buf.empty());
}

TEST(CompileResourceSnapshot, ReplaysTheParsedFetchProgramWithoutReadingLiveGuestCode) {
    FetchShaderData generation_a{};
    generation_a.size = 24;
    generation_a.vertex_offset_sgpr = 6;
    generation_a.instance_offset_sgpr = 8;
    generation_a.attributes.push_back(VertexAttribute{
        .semantic = 3,
        .dest_vgpr = 5,
        .num_elements = 4,
        .sgpr_base = 2,
        .dword_offset = 12,
        .instance_data = VertexAttribute::Plain,
        .inst_offset = 16,
        .data_format = 11,
        .num_format = 4,
    });

    Info compiling{};
    compiling.has_fetch_shader = true;
    compiling.fetch_shader_sgpr_base = 0;
    compiling.UseCompileFetchShaderSnapshot(generation_a);

    const auto replayed = ParseFetchShader(compiling);

    ASSERT_TRUE(replayed.has_value());
    EXPECT_EQ(*replayed, generation_a);
    ASSERT_EQ(replayed->attributes.size(), 1u);
    EXPECT_EQ(replayed->attributes[0].inst_offset, 16u);
    EXPECT_EQ(replayed->attributes[0].data_format, 11u);
    EXPECT_EQ(replayed->attributes[0].num_format, 4u);
}

TEST(CompileResourceSnapshot, CachedPermutationCompileConsumesTheSpecializationGeneration) {
    std::ifstream stream{U3_PIPELINE_CACHE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const std::string source{std::istreambuf_iterator<char>{stream},
                             std::istreambuf_iterator<char>{}};

    const auto get_program = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto next_function = source.find("std::optional<vk::ShaderModule>", get_program);
    ASSERT_NE(get_program, std::string::npos);
    ASSERT_NE(next_function, std::string::npos);
    const auto body = source.substr(get_program, next_function - get_program);

    const auto refresh = body.find("info.RefreshFlatBuf();");
    const auto capture = body.find("CompileResourceSnapshot::Capture(info)", refresh);
    const auto specialize = body.find("auto spec = Shader::StageSpecialization(", capture);
    const auto fetch = body.find("spec.fetch_shader_data", specialize);
    const auto new_info = body.find("auto new_info = Shader::Info(", fetch);
    const auto compile = body.find("module = CompileModule(new_info", new_info);
    ASSERT_NE(refresh, std::string::npos);
    ASSERT_NE(capture, std::string::npos);
    ASSERT_NE(specialize, std::string::npos);
    ASSERT_NE(fetch, std::string::npos);
    ASSERT_NE(new_info, std::string::npos);
    ASSERT_NE(compile, std::string::npos);
    EXPECT_LT(refresh, capture);
    EXPECT_LT(capture, specialize);
    EXPECT_LT(specialize, fetch);
    EXPECT_LT(fetch, compile);
    EXPECT_NE(body.substr(compile, 320).find("&compile_resources"), std::string::npos);
    EXPECT_NE(body.substr(compile, 320).find("compile_fetch_shader"), std::string::npos);

    const auto compile_function = source.find("vk::ShaderModule PipelineCache::CompileModule(");
    const auto translate = source.find("Shader::TranslateProgram(", compile_function);
    const auto use_resources =
        source.find("info.UseCompileResourceSnapshot(*compile_resources)", compile_function);
    const auto use_fetch =
        source.find("info.UseCompileFetchShaderSnapshot(*compile_fetch_shader)", compile_function);
    ASSERT_NE(compile_function, std::string::npos);
    ASSERT_NE(translate, std::string::npos);
    ASSERT_NE(use_resources, std::string::npos);
    ASSERT_NE(use_fetch, std::string::npos);
    EXPECT_LT(use_resources, translate);
    EXPECT_LT(use_fetch, translate);
}

TEST(CompileResourceSnapshot, CompileOnlyStateIsNotPartOfThePersistentShaderCacheAbi) {
    static_assert(std::is_base_of_v<InfoPersistent, Info>);
    static_assert(!HasCompileFlattenedSnapshot<InfoPersistent>);
    static_assert(!HasCompileFetchSnapshot<InfoPersistent>);
    SUCCEED();
}

TEST(VertexInputSnapshot, CapturesOneDescriptorGenerationForSpecializationPipelineAndBinding) {
    FetchShaderData fetch{};
    fetch.attributes.push_back(VertexAttribute{.semantic = 7});
    fetch.attributes.push_back(VertexAttribute{.semantic = 2});

    const std::array generation_a{
        MakeVertexBuffer(0x1000, 16, 64, 4),
        MakeVertexBuffer(0x2000, 32, 128, 11),
    };
    const std::array generation_b{
        MakeVertexBuffer(0x9000, 4, 8, 1),
        MakeVertexBuffer(0xa000, 8, 16, 2),
    };
    auto live = generation_a;
    size_t reads{};

    const auto snapshot = VertexInputSnapshot::Capture(fetch, [&](const VertexAttribute&) {
        const auto index = reads++;
        const auto captured = live[index];
        live[index] = generation_b[index];
        return captured;
    });

    ASSERT_EQ(reads, generation_a.size());
    ASSERT_EQ(snapshot.buffers.size(), generation_a.size());
    for (size_t phase = 0; phase < 3; ++phase) {
        EXPECT_EQ(snapshot.buffers[0], generation_a[0]);
        EXPECT_EQ(snapshot.buffers[1], generation_a[1]);
    }
    EXPECT_EQ(live, generation_b);
}

TEST(VertexInputSnapshot, CurrentDrawCarriesTheSameSnapshotThroughEveryConsumer) {
    const auto cache = ReadSource(U3_PIPELINE_CACHE_SOURCE_PATH);
    const auto graphics = ReadSource(U3_GRAPHICS_PIPELINE_SOURCE_PATH);
    const auto rasterizer = ReadSource(U3_RASTERIZER_SOURCE_PATH);
    const auto buffers = ReadSource(U3_BUFFER_CACHE_SOURCE_PATH);

    const auto refresh_stages = cache.find("bool PipelineCache::RefreshGraphicsStages()");
    const auto refresh_compute = cache.find("bool PipelineCache::RefreshComputeKey()", refresh_stages);
    ASSERT_NE(refresh_stages, std::string::npos);
    ASSERT_NE(refresh_compute, std::string::npos);
    const auto refresh_body = cache.substr(refresh_stages, refresh_compute - refresh_stages);
    EXPECT_NE(refresh_body.find("vertex_inputs = {}"), std::string::npos);
    EXPECT_NE(refresh_body.find("Shader::Gcn::VertexInputSnapshot vertex_inputs_"),
              std::string::npos);
    EXPECT_NE(refresh_body.find("vertex_inputs = std::move(vertex_inputs_)"), std::string::npos);
    EXPECT_NE(refresh_body.find("vertex_inputs.buffers"), std::string::npos);
    EXPECT_EQ(refresh_body.find("attrib.GetSharp(*vs_info)"), std::string::npos);

    const auto get_graphics = cache.find("PipelineCache::GraphicsPipelineResult "
                                         "PipelineCache::GetGraphicsPipeline()");
    const auto get_compute = cache.find("const ComputePipeline* PipelineCache::GetComputePipeline()",
                                        get_graphics);
    ASSERT_NE(get_graphics, std::string::npos);
    ASSERT_NE(get_compute, std::string::npos);
    const auto graphics_cache_body = cache.substr(get_graphics, get_compute - get_graphics);
    EXPECT_NE(graphics_cache_body.find("vertex_inputs"), std::string::npos);
    EXPECT_NE(graphics_cache_body.find("return {it->second.get(), std::move(vertex_inputs)}"),
              std::string::npos);

    const auto get_program = cache.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto replace_shader = cache.find("std::optional<vk::ShaderModule>", get_program);
    ASSERT_NE(get_program, std::string::npos);
    ASSERT_NE(replace_shader, std::string::npos);
    const auto program_body = cache.substr(get_program, replace_shader - get_program);
    EXPECT_NE(program_body.find("profile, start, &vertex_inputs"), std::string::npos);
    EXPECT_NE(program_body.find("profile, binding, &vertex_inputs"), std::string::npos);

    const auto direct = rasterizer.find("void Rasterizer::Draw(bool is_indexed");
    const auto indirect = rasterizer.find("void Rasterizer::DrawIndirect(", direct);
    const auto dispatch = rasterizer.find("void Rasterizer::DispatchDirect()", indirect);
    ASSERT_NE(direct, std::string::npos);
    ASSERT_NE(indirect, std::string::npos);
    ASSERT_NE(dispatch, std::string::npos);
    for (const auto draw_body : {rasterizer.substr(direct, indirect - direct),
                                 rasterizer.substr(indirect, dispatch - indirect)}) {
        EXPECT_NE(draw_body.find("const auto [pipeline, vertex_inputs] = "
                                 "pipeline_cache.GetGraphicsPipeline()"),
                  std::string::npos);
        EXPECT_NE(draw_body.find("BindVertexBuffers(*pipeline, vertex_inputs"), std::string::npos);
    }

    const auto bind = buffers.find("void BufferCache::BindVertexBuffers(");
    ASSERT_NE(bind, std::string::npos);
    const auto bind_body = buffers.substr(bind, 900);
    EXPECT_NE(bind_body.find("const Shader::Gcn::VertexInputSnapshot& vertex_inputs"),
              std::string::npos);
    EXPECT_NE(bind_body.find("guest_buffers, vertex_inputs"), std::string::npos);

    const auto get_inputs = graphics.find("void GraphicsPipeline::GetVertexInputs(");
    ASSERT_NE(get_inputs, std::string::npos);
    const auto input_body = graphics.substr(get_inputs, 2600);
    EXPECT_NE(input_body.find("const Shader::Gcn::VertexInputSnapshot& vertex_inputs"),
              std::string::npos);
    EXPECT_NE(input_body.find("vertex_inputs.buffers[index]"), std::string::npos);
    EXPECT_EQ(input_body.find("attrib.GetSharp("), std::string::npos);
}

} // namespace
} // namespace Shader::Gcn
