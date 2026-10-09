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

TEST(SrtMemorySyncSource, SnapshotReadNeverPublishesIntoGuestBackingOrChangesOwnership) {
    const auto source = ReadBufferCacheSource();
    const auto begin = source.find("bool BufferCache::ReadMemorySnapshot");
    const auto end = source.find("template <bool async, bool exact>", begin);

    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_NE(body.find("ReadAuthoritativeMemorySnapshot"), std::string::npos);
    EXPECT_NE(body.find("DownloadBufferMemorySnapshot"), std::string::npos);
    EXPECT_EQ(body.find("SynchronizeBuffer("), std::string::npos);
    EXPECT_EQ(body.find("TryWriteBacking"), std::string::npos);
    EXPECT_EQ(body.find("UnmarkRegionAsGpuModified"), std::string::npos);
    EXPECT_EQ(body.find("gpu_modified_ranges.Subtract"), std::string::npos);
    EXPECT_EQ(body.find("FindBuffer("), std::string::npos);
}

TEST(SrtMemorySyncSource, PrivateGpuSnapshotOverlaysExactRangesWithoutMutatingOwnership) {
    const auto source = ReadBufferCacheSource();
    const auto begin = source.find("bool BufferCache::DownloadBufferMemorySnapshot");
    const auto end = source.find("BufferCache::PreparedVertexBuffers", begin);

    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_NE(body.find("gpu_modified_ranges.ForEachInRange"), std::string::npos);
    EXPECT_NE(body.find("CopySparseMemory"), std::string::npos);
    EXPECT_NE(body.find("ApplyMemorySnapshotOverlays"), std::string::npos);
    const auto capture = body.find("memory->CopySparseMemory(cpu_capture_address");
    // The composition result may be passed to a diagnostic selector before returning;
    // protect the actual overlay operation and ordering, not its old return spelling.
    const auto gpu_overlay = body.find("ApplyMemorySnapshotOverlays(");
    const auto cpu_overlay = body.find("cpu_page_write_tracker.ApplyPendingWrites(");
    ASSERT_NE(capture, std::string::npos);
    ASSERT_NE(gpu_overlay, std::string::npos);
    ASSERT_NE(cpu_overlay, std::string::npos);
    EXPECT_LT(capture, gpu_overlay);
    EXPECT_LT(gpu_overlay, cpu_overlay);
    EXPECT_NE(body.find("PlanCpuSnapshotCapture(device_addr, size)"), std::string::npos);
    EXPECT_EQ(body.find("SynchronizeBuffer("), std::string::npos);
    EXPECT_EQ(body.find("gpu_modified_ranges.Subtract"), std::string::npos);
    EXPECT_EQ(body.find("TryWriteBacking"), std::string::npos);
    EXPECT_EQ(body.find("UnmarkRegionAsGpuModified"), std::string::npos);
    EXPECT_EQ(body.find("FindBuffer("), std::string::npos);
}

} // namespace
} // namespace VideoCore
