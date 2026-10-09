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
    EXPECT_NE(body.find("AllPagesOwnedBy("), std::string::npos);
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

TEST(SrtMemorySyncSource, NormalReadbackUsesDirtyRangeOwnerForEveryCopy) {
    const auto source = ReadBufferCacheSource();
    const auto begin = source.find("template <bool async, bool exact>");
    const auto end = source.find("bool BufferCache::DownloadBufferMemorySnapshot", begin);

    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_NE(body.find("PlanOwnedDownloadRanges<DownloadOwner>"), std::string::npos);
    EXPECT_NE(body.find("slot_buffers[owner.buffer_id]"), std::string::npos);
    EXPECT_NE(body.find("copy.buffer->buffer"), std::string::npos);
    EXPECT_NE(body.find("copy.address"), std::string::npos);
    EXPECT_EQ(body.find("cmdbuf.copyBuffer(buffer.buffer"), std::string::npos);
}

TEST(SrtMemorySyncSource, SnapshotOverlayProbeIsOptInAndReportsSourceAccounting) {
    const auto source = ReadBufferCacheSource();
    const auto begin = source.find("bool BufferCache::DownloadBufferMemorySnapshot");
    const auto end = source.find("BufferCache::PreparedVertexBuffers", begin);

    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_NE(source.find("SHADPS4_SRT_SNAPSHOT_OVERLAY_DIAGNOSTICS"), std::string::npos);
    EXPECT_NE(body.find("SrtSnapshotOverlayPlanV1"), std::string::npos);
    EXPECT_NE(body.find("SrtSnapshotOverlaySourceV1"), std::string::npos);
    EXPECT_NE(body.find("buffer_id"), std::string::npos);
    EXPECT_NE(body.find("source_offset"), std::string::npos);
    EXPECT_NE(body.find("download_offset"), std::string::npos);
    EXPECT_NE(body.find("copy.region.dstOffset += offset"), std::string::npos);
    EXPECT_NE(body.find("receipt_request != nullptr"), std::string::npos);
}

TEST(SrtMemorySyncSource, SnapshotOverlayProbeCorrelatesFocusAndOwnerGeneration) {
    const auto source = ReadBufferCacheSource();
    const auto begin = source.find("bool BufferCache::DownloadBufferMemorySnapshot");
    const auto end = source.find("BufferCache::PreparedVertexBuffers", begin);

    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    // A focus-only diagnostic must identify the exact snapshot and allocation incarnation that
    // produced each overlay. Without this correlation, a capped log cannot distinguish a stale
    // source buffer from a legitimate GPU write at the same guest address.
    EXPECT_NE(body.find("diagnostic_snapshot_id"), std::string::npos);
    EXPECT_NE(body.find("owner_generation"), std::string::npos);
    EXPECT_NE(body.find("focus_address"), std::string::npos);
    EXPECT_NE(body.find("focus_size"), std::string::npos);
    EXPECT_NE(body.find("focus_intersects"), std::string::npos);
    EXPECT_NE(body.find("SrtSnapshotOverlayPlanV1"), std::string::npos);
    EXPECT_NE(body.find("SrtSnapshotOverlaySourceV1"), std::string::npos);
}

TEST(SrtMemorySyncSource, SnapshotReadbackSuppressesReentrantRetirementAcrossStagingMap) {
    const auto source = ReadBufferCacheSource();
    const auto begin = source.find("bool BufferCache::DownloadBufferMemorySnapshot");
    const auto end = source.find("BufferCache::PreparedVertexBuffers", begin);

    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    const auto guard = body.find("ScopedPendingOperationDrainSuppression");
    const auto map = body.find("download_buffer.Map");
    const auto finish = body.find("CompletePrivateSnapshotReadback");
    ASSERT_NE(map, std::string::npos);
    ASSERT_NE(finish, std::string::npos);
    // The source Buffer* list is built before Map. A stream-wrap wait inside Map can otherwise
    // pop a deferred erase and invalidate those pointers before the copy commands are recorded.
    EXPECT_NE(guard, std::string::npos);
    EXPECT_LT(guard, map);
    EXPECT_LT(guard, finish);
}

TEST(SrtMemorySyncSource, CompositionReceiptObservesFinalPrivateBytesBeforeDiagnosticSelection) {
    const auto source = ReadBufferCacheSource();
    const auto begin = source.find("bool BufferCache::DownloadBufferMemorySnapshot");
    const auto end = source.find("BufferCache::PreparedVertexBuffers", begin);

    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    const auto capture_cpu = body.find("memory->CopySparseMemory(cpu_capture_address");
    const auto gpu_overlay = body.find("ApplyMemorySnapshotOverlays(");
    const auto cpu_overlay = body.find("cpu_page_write_tracker.ApplyPendingWrites(");
    const auto receipt = body.find("CaptureSrtSnapshotCompositionReceipt(");
    const auto diagnostic_selection = body.find("GetSrtCpuSnapshotSelection().Complete(");
    ASSERT_NE(capture_cpu, std::string::npos);
    ASSERT_NE(gpu_overlay, std::string::npos);
    ASSERT_NE(cpu_overlay, std::string::npos);
    ASSERT_NE(receipt, std::string::npos);
    ASSERT_NE(diagnostic_selection, std::string::npos);
    EXPECT_LT(capture_cpu, gpu_overlay);
    EXPECT_LT(gpu_overlay, cpu_overlay);
    EXPECT_LT(cpu_overlay, receipt);
    EXPECT_LT(receipt, diagnostic_selection);
    EXPECT_NE(body.find("device_addr - cpu_capture_address"), std::string::npos);
    EXPECT_NE(body.find("overlay_bytes"), std::string::npos);
    EXPECT_NE(body.find("overlay.size"), std::string::npos);
    EXPECT_NE(body.find("if (normal_completed"), std::string::npos);
    EXPECT_EQ(body.find("TryWriteBacking"), std::string::npos);
    EXPECT_EQ(body.find("UnmarkRegionAsGpuModified"), std::string::npos);
    EXPECT_EQ(body.find("gpu_modified_ranges.Subtract"), std::string::npos);
}

} // namespace
} // namespace VideoCore
