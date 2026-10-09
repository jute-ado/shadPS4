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

void PS4_SYSV_ABI CountingWalker(const u32* user_data, u32* flattened,
                                 PFN_SrtRangeRead, void*) {
    ++walker_calls;
    std::ranges::copy_n(user_data, NUM_USER_DATA_REGS, flattened);
}

template <typename T>
concept HasCompileFlattenedSnapshot = requires(T value) {
    value.compile_flattened_ud_snapshot;
};

template <typename T>
concept HasCompileFetchSnapshot = requires(T value) {
    value.compile_fetch_shader_snapshot;
};

TEST(CompileResourceSnapshot, ReplaysSelectedDescriptorGenerationWithoutCallingLiveWalker) {
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

TEST(CompileResourceSnapshot, RejectsMalformedDescriptorImageBeforeReplay) {
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

TEST(CompileResourceSnapshot, ReplaysParsedFetchProgramWithoutReadingLiveGuestCode) {
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
}

TEST(CompileResourceSnapshot, CachedPermutationCompileConsumesSelectedGeneration) {
    std::ifstream stream{SHADPS4_PIPELINE_CACHE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const std::string source{std::istreambuf_iterator<char>{stream},
                             std::istreambuf_iterator<char>{}};

    const auto get_program = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto next_function = source.find(
        "std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(", get_program);
    ASSERT_NE(get_program, std::string::npos);
    ASSERT_NE(next_function, std::string::npos);
    const auto body = source.substr(get_program, next_function - get_program);

    const auto session = body.find("SrtReadSession srt_read_session");
    const auto refresh = body.find(
        "info.RefreshFlatBuf(MakeSrtMemoryReader(srt_read_session))", session);
    const auto capture = body.find("CompileResourceSnapshot::Capture(info)", refresh);
    const auto specialize = body.find("auto spec = Shader::StageSpecialization(", capture);
    const auto compile = body.find("module = CompileModule(new_info", specialize);
    ASSERT_NE(session, std::string::npos);
    ASSERT_NE(refresh, std::string::npos);
    ASSERT_NE(capture, std::string::npos);
    ASSERT_NE(specialize, std::string::npos);
    ASSERT_NE(compile, std::string::npos);
    EXPECT_LT(session, refresh);
    EXPECT_LT(refresh, capture);
    EXPECT_LT(capture, specialize);
    EXPECT_LT(specialize, compile);
    EXPECT_NE(body.substr(compile, 400).find("&compile_resources"), std::string::npos);
    EXPECT_NE(body.substr(compile, 400).find("compile_fetch_shader"), std::string::npos);

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

TEST(CompileResourceSnapshot, CompileOnlyStateIsExcludedFromPersistentShaderCacheAbi) {
    static_assert(std::is_base_of_v<InfoPersistent, Info>);
    static_assert(!HasCompileFlattenedSnapshot<InfoPersistent>);
    static_assert(!HasCompileFetchSnapshot<InfoPersistent>);
    SUCCEED();
}

} // namespace
} // namespace Shader::Gcn
