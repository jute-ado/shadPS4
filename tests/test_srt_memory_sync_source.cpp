// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace VideoCore {
namespace {

std::string ReadBufferCacheSource() {
    std::ifstream file{SHADPS4_BUFFER_CACHE_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

TEST(SrtMemorySyncSource, ExactReadbackNeverCreatesOrReplacesBufferTopology) {
    const auto source = ReadBufferCacheSource();
    const auto begin = source.find("bool BufferCache::ReadMemoryExact");
    const auto end = source.find("void BufferCache::DownloadBufferMemory", begin);

    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_EQ(body.find("FindBuffer("), std::string::npos);
    EXPECT_NE(body.find("page_table["), std::string::npos);
    EXPECT_NE(body.find("IsBufferInvalid"), std::string::npos);
    EXPECT_NE(body.find("IsInBounds"), std::string::npos);
    EXPECT_NE(body.find("DownloadBufferMemory<false, true>"), std::string::npos);
}

} // namespace
} // namespace VideoCore
