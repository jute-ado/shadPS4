// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace Vulkan {
namespace {

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

TEST(HostCompilationPause, RuntimeCompilationNeverSuspendsGuestThreads) {
    const auto source = ReadSource(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    ASSERT_FALSE(source.empty());
    EXPECT_EQ(source.find("host_compilation_pause.h"), std::string::npos);
    EXPECT_EQ(source.find("ScopedHostCompilationGuestPause"), std::string::npos);
    EXPECT_EQ(source.find("PauseGuestThreads"), std::string::npos);
}

TEST(PersistentShaderCache, BackendSemanticChangesInvalidateStoredSpirv) {
    const auto source = ReadSource(SHADPS4_PIPELINE_SERIALIZATION_SOURCE_PATH);

    ASSERT_FALSE(source.empty());
    EXPECT_NE(source.find("static constexpr u32 ShaderBinaryVersion = 4u;"),
              std::string::npos);
}

} // namespace
} // namespace Vulkan
