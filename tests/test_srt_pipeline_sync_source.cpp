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

TEST(SrtMemorySyncSource, PipelineCompilationAndRefreshUseBoundedPrivateSnapshotSessions) {
    const auto pipeline = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto rasterizer = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);

    EXPECT_NE(pipeline.find("using SrtReadSession ="), std::string::npos);
    EXPECT_NE(pipeline.find("SrtReadSession srt_read_session"), std::string::npos);
    EXPECT_NE(pipeline.find("MakeSrtMemoryReader(srt_read_session)"), std::string::npos);
    EXPECT_NE(pipeline.find("Shader::TranslateProgram("), std::string::npos);
    EXPECT_NE(pipeline.find("info.RefreshFlatBuf(MakeSrtMemoryReader(srt_read_session))"),
              std::string::npos);
    EXPECT_EQ(pipeline.find("return {&ReadSrtRange, this};"), std::string::npos);
    EXPECT_NE(rasterizer.find(
                  "pipeline_cache{instance, scheduler, liverpool, page_manager, buffer_cache}"),
              std::string::npos);
}

} // namespace
} // namespace VideoCore
