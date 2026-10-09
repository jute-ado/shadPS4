// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

// Production connections are tested separately from the pure resolver so enabled behavior
// cannot be inferred from isolated policy tests.
namespace {

std::string ReadRendererSource(const char* filename) {
    const auto path =
        std::filesystem::path{SHADPS4_INDEX_PROVENANCE_RASTERIZER_SOURCE}.parent_path() / filename;
    std::ifstream input{path, std::ios::binary};
    EXPECT_TRUE(input.is_open());
    std::string text{std::istreambuf_iterator<char>{input}, {}};
    std::erase(text, '\r');
    return text;
}

std::string Section(const std::string& text, const char* begin_marker, const char* end_marker) {
    const auto begin = text.find(begin_marker);
    EXPECT_NE(begin, std::string::npos);
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = text.find(end_marker, begin + 1);
    EXPECT_NE(end, std::string::npos);
    return end == std::string::npos ? std::string{} : text.substr(begin, end - begin);
}

TEST(IndirectDrawContextWiring, OnlyIndirectDrawSuppliesPacketContext) {
    const auto text = ReadRendererSource("vk_rasterizer.cpp");
    const auto direct = Section(text, "void Rasterizer::Draw(", "void Rasterizer::DrawIndirect(");
    const auto indirect =
        Section(text, "void Rasterizer::DrawIndirect(", "void Rasterizer::DispatchDirect(");
    EXPECT_NE(direct.find("pipeline_cache.GetGraphicsPipeline()"), std::string::npos);
    EXPECT_NE(indirect.find("pipeline_cache.GetGraphicsPipeline(register_locations)"),
              std::string::npos);
}

TEST(IndirectDrawContextWiring, ImmutableMappingReachesEverySelectionBoundary) {
    const auto text = ReadRendererSource("vk_pipeline_cache.cpp");
    const auto entry = Section(text, "const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(",
                               "const ComputePipeline* PipelineCache::GetComputePipeline(");
    const auto resolve = entry.find("ResolveVertexDrawUserData(");
    const auto refresh = entry.find("RefreshGraphicsKey(");
    ASSERT_NE(resolve, std::string::npos);
    ASSERT_NE(refresh, std::string::npos);
    EXPECT_LT(resolve, refresh);
    EXPECT_NE(entry.find("ShaderStageEnable::VgtStages::Vs"), std::string::npos);
    const auto graphics_key = Section(text, "bool PipelineCache::RefreshGraphicsKey(",
                                      "bool PipelineCache::RefreshGraphicsStages(");
    EXPECT_NE(graphics_key.find("RefreshGraphicsStages(draw_user_data)"), std::string::npos);
    const auto graphics_stages = Section(text, "bool PipelineCache::RefreshGraphicsStages(",
                                         "bool PipelineCache::RefreshComputeKey(");
    EXPECT_NE(
        graphics_stages.find("GetProgram(stage_in, stage_out, params, binding, draw_user_data)"),
        std::string::npos);
    EXPECT_NE(text.find("BuildRuntimeInfo(stage, l_stage, reader, draw_user_data)"),
              std::string::npos);
}

TEST(IndirectDrawContextWiring, RuntimeAwareProgramKeyPrecedesEveryCachedWalkerRefresh) {
    const auto text = ReadRendererSource("vk_pipeline_cache.cpp");
    const auto begin = text.find("PipelineCache::Result PipelineCache::GetProgram(");
    ASSERT_NE(begin, std::string::npos);
    const auto body = text.substr(begin);
    const auto runtime = body.find("BuildRuntimeInfo(stage, l_stage, reader, draw_user_data)");
    const auto key = body.find("Shader::MakeProgramCacheKey(");
    const auto lookup = body.find("program_cache.try_emplace(cache_key)");
    const auto walker = body.find("aggregate_snapshot.RefreshFlatBuf(reader)");
    ASSERT_NE(runtime, std::string::npos);
    ASSERT_NE(key, std::string::npos);
    ASSERT_NE(lookup, std::string::npos);
    ASSERT_NE(walker, std::string::npos);
    EXPECT_LT(runtime, key);
    EXPECT_LT(key, lookup);
    EXPECT_LT(lookup, walker);
    EXPECT_NE(body.substr(runtime, key - runtime).find("runtime_info.vs_info.draw_user_data"),
              std::string::npos);
}

TEST(IndirectDrawContextWiring, PreloadedProgramKeyUsesStoredRuntimeMapping) {
    const auto text = ReadRendererSource("vk_pipeline_serialization.cpp");
    const auto body =
        Section(text, "bool PipelineCache::LoadPipelineStage(", "void PipelineCache::WarmUp(");
    const auto key = body.find("Shader::MakeProgramCacheKey(");
    const auto lookup = body.find("program_cache.find(cache_key)");
    ASSERT_NE(key, std::string::npos);
    ASSERT_NE(lookup, std::string::npos);
    EXPECT_LT(key, lookup);
    const auto runtime = body.find("spec.runtime_info.vs_info.draw_user_data");
    ASSERT_NE(runtime, std::string::npos);
    EXPECT_LT(runtime, key);
    EXPECT_EQ(body.find("register_locations"), std::string::npos);
    EXPECT_EQ(body.find("liverpool->regs"), std::string::npos);
}

} // namespace
