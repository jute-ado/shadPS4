// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace {

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

TEST(PipelineCompilationPolicy, DoesNotSuspendGuestThreadsDuringHostCompilation) {
    const auto source = ReadSource(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    ASSERT_FALSE(source.empty());
    EXPECT_EQ(source.find("ScopedHostCompilationGuestPause"), std::string::npos);
    EXPECT_EQ(source.find("PauseGuestThreads("), std::string::npos);
    EXPECT_EQ(source.find("ResumeGuestThreads("), std::string::npos);
}

} // namespace
