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

TEST(SrtMemorySyncSource, SrtGateUsesBytePreciseGpuWriteTracking) {
    const auto source = ReadBufferCacheSource();
    EXPECT_NE(source.find("bool BufferCache::IsRegionGpuModifiedExact"), std::string::npos);
    EXPECT_NE(source.find("gpu_modified_ranges.Intersects(addr, size)"), std::string::npos);
}

std::string ReadRasterizerSource() {
    std::ifstream file{SHADPS4_RASTERIZER_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

TEST(SrtMemorySyncSource, SnapshotDoesNotPublishIntoGuestMemoryOrTransferOwnership) {
    const auto source = ReadBufferCacheSource();
    const auto begin = source.find("bool BufferCache::DownloadBufferMemorySnapshot");
    const auto end = source.find("BufferCache::PreparedVertexBuffers", begin);

    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_EQ(body.find("TryWriteBacking"), std::string::npos);
    EXPECT_EQ(body.find("gpu_modified_ranges.Subtract"), std::string::npos);
    EXPECT_EQ(body.find("UnmarkRegionAsGpuModified"), std::string::npos);
    EXPECT_EQ(body.find("gpu_modified_ranges.ForEachInRange"), std::string::npos);
    EXPECT_NE(body.find(".srcOffset = device_addr - buffer.CpuAddr()"), std::string::npos);
    EXPECT_NE(body.find(".size = size"), std::string::npos);
    EXPECT_NE(body.find("std::memcpy(snapshot"), std::string::npos);
}

TEST(SrtMemorySyncSource, GpuSnapshotFirstUploadsCpuOwnedBytesIntoContainingBuffer) {
    const auto source = ReadBufferCacheSource();
    const auto begin = source.find("bool BufferCache::ReadMemorySnapshot");
    const auto end = source.find("template <bool async, bool exact>", begin);

    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    const auto synchronize = body.find("SynchronizeBuffer(buffer, device_addr");
    const auto snapshot = body.find("DownloadBufferMemorySnapshot(buffer, device_addr");
    ASSERT_NE(synchronize, std::string::npos);
    ASSERT_NE(snapshot, std::string::npos);
    EXPECT_LT(synchronize, snapshot);
}

TEST(SrtMemorySyncSource, UnmappedBufferDescriptorIsRejectedBeforeCacheRegistration) {
    const auto source = ReadRasterizerSource();
    const auto begin = source.find("void Rasterizer::BindBuffers(");
    const auto end = source.find("void Rasterizer::BindTextures(", begin);

    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    const auto mapping_check = body.find("VideoCore::IsGuestBufferBindingMapped(");
    const auto find_buffer = body.find("buffer_cache.FindBuffer(");
    ASSERT_NE(mapping_check, std::string::npos);
    ASSERT_NE(find_buffer, std::string::npos);
    EXPECT_LT(mapping_check, find_buffer);
}

} // namespace
} // namespace VideoCore
