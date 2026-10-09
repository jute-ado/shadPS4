// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace Vulkan {
namespace {

std::string ReadSource(const char* path) {
    std::ifstream file{path, std::ios::binary};
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

TEST(PipelinePermutationInfoSource, RetainsResourceLayoutForEveryCompiledPermutation) {
    const auto header = ReadSource(SHADPS4_PIPELINE_CACHE_HEADER_PATH);
    const auto source = ReadSource(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    EXPECT_NE(header.find("std::unique_ptr<Shader::Info> info"), std::string::npos);
    EXPECT_NE(source.find("std::make_unique<Shader::Info>(stage, l_stage, params)"),
              std::string::npos);
    EXPECT_NE(source.find("selected_info.AddBindings(binding)"), std::string::npos);
    EXPECT_NE(source.find("return std::make_tuple(&selected_info"), std::string::npos);
}

TEST(PipelinePermutationInfoSource, DepthTargetRefreshesAReplacedCachedImageBeforeBinding) {
    const auto source = ReadSource(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto begin = source.find("RenderState Rasterizer::BeginRendering");
    const auto end = source.find("void Rasterizer::UpdateDynamicState", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    const auto stale = body.find("texture_cache.GetImage(db_desc.first).binding.needs_rebind");
    const auto refresh = body.find("db_desc.first = bound_images.emplace_back(", stale);
    const auto find = body.find("texture_cache.FindImage(db_desc.second)", refresh);
    ASSERT_NE(stale, std::string::npos);
    ASSERT_NE(refresh, std::string::npos);
    ASSERT_NE(find, std::string::npos);
    EXPECT_LT(stale, refresh);
    EXPECT_LT(refresh, find);
}

} // namespace
} // namespace Vulkan
