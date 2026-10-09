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

void PS4_SYSV_ABI CountingWalker(const u32* user_data, u32* flattened, PFN_SrtRangeRead, void*) {
    ++walker_calls;
    std::ranges::copy_n(user_data, NUM_USER_DATA_REGS, flattened);
}

template <typename T>
concept HasCompileFlattenedSnapshot = requires(T value) {
    value.compile_flattened_ud_snapshot;
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

    Info compiling{};
    compiling.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS;
    compiling.srt_info.walker_func = CountingWalker;
    compiling.user_data = generation_b;
    compiling.UseCompileResourceSnapshot(snapshot);
    walker_calls = 0;
    compiling.RefreshFlatBuf();

    EXPECT_EQ(walker_calls, 0u);
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

TEST(CompileResourceSnapshot, CachedPermutationCompileConsumesSelectedGeneration) {
    std::ifstream stream{SHADPS4_PIPELINE_CACHE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const std::string source{std::istreambuf_iterator<char>{stream},
                             std::istreambuf_iterator<char>{}};
    const auto get_program = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto compile = source.find("module = CompileModule(new_info", get_program);
    const auto capture = source.find("CompileResourceSnapshot::Capture(info)", get_program);
    ASSERT_NE(get_program, std::string::npos);
    ASSERT_NE(compile, std::string::npos);
    ASSERT_NE(capture, std::string::npos);
    EXPECT_LT(capture, compile);
    EXPECT_NE(source.substr(compile, 400).find("&compile_resources"), std::string::npos);
}

TEST(CompileResourceSnapshot, CompileOnlyStateIsExcludedFromPersistentShaderCacheAbi) {
    static_assert(std::is_base_of_v<InfoPersistent, Info>);
    static_assert(!HasCompileFlattenedSnapshot<InfoPersistent>);
    SUCCEED();
}

} // namespace
} // namespace Shader::Gcn
