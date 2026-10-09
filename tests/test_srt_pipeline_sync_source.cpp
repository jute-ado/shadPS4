// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace VideoCore {
namespace {

std::string ReadText(const char* path) {
    std::ifstream file{path, std::ios::binary};
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

TEST(SrtMemorySyncSource, PipelineCompilationAndRefreshUseExactSynchronizer) {
    const auto pipeline = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto rasterizer = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);

    EXPECT_NE(pipeline.find("GetSrtMemorySynchronizer()"), std::string::npos);
    EXPECT_NE(pipeline.find("return {&SynchronizeSrtRange, this};"), std::string::npos);
    EXPECT_NE(pipeline.find("SynchronizeSrtMemoryRange(cache.page_manager, cache.buffer_cache"),
              std::string::npos);
    EXPECT_NE(pipeline.find("Shader::TranslateProgram(code, pools, info, runtime_info, profile,"),
              std::string::npos);
    EXPECT_NE(pipeline.find("GetSrtMemorySynchronizer());"), std::string::npos);
    EXPECT_NE(pipeline.find("info.RefreshFlatBuf(GetSrtMemorySynchronizer())"), std::string::npos);
    EXPECT_NE(rasterizer.find(
                  "pipeline_cache{instance, scheduler, liverpool, page_manager, buffer_cache}"),
              std::string::npos);
}

} // namespace
} // namespace VideoCore
