// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <sstream>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/srt_memory_sync.h"

namespace VideoCore {
namespace {

using Route = SrtSnapshotCompositionRoute;

constexpr SrtSnapshotCompositionReceipt CpuReceipt{
    .snapshot_address = 0x4000,
    .snapshot_size = 0x1000,
    .focus_address = 0x4020,
    .focus_size = 0x20,
    .route = Route::CpuOnly,
    .sample_size = 0x10,
};

constexpr SrtSnapshotCompositionReceipt GpuReceipt{
    .snapshot_address = 0x8000,
    .snapshot_size = 0x1000,
    .focus_address = 0x8040,
    .focus_size = 0x20,
    .route = Route::GpuComposed,
    .sample_size = 0x10,
    .overlay_count = 1,
    .overlay_bytes = 0x40,
};

TEST(SrtSourceRead, CapturesBoundedCpuBytesAndRange) {
    constexpr std::array<u8, 32> bytes{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                       0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
                                       0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                                       0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f};
    const auto record = CaptureSrtSourceReadRecord(0x4020, 32, SrtMemoryReadFailure::None, false,
                                                   bytes, CpuReceipt);

    EXPECT_TRUE(record.request_known);
    EXPECT_EQ(record.request_address, 0x4020U);
    EXPECT_EQ(record.request_size, 32U);
    EXPECT_EQ(record.failure, SrtMemoryReadFailure::None);
    EXPECT_EQ(record.route, Route::CpuOnly);
    EXPECT_EQ(record.snapshot_address, 0x4000U);
    EXPECT_EQ(record.snapshot_size, 0x1000U);
    EXPECT_TRUE(record.source_bytes_known);
    EXPECT_EQ(record.bytes_captured, SrtSourceReadRecord::MaximumBytes);
    EXPECT_TRUE(std::equal(bytes.begin(), bytes.begin() + SrtSourceReadRecord::MaximumBytes,
                           record.first_bytes.begin()));
}

TEST(SrtSourceRead, PreservesDiagnosticSnapshotIdFromCompositionReceipt) {
    constexpr std::array<u8, 8> bytes{1, 2, 3, 4, 5, 6, 7, 8};
    const SrtSnapshotCompositionReceipt receipt{
        .snapshot_address = 0x4000,
        .snapshot_size = 0x1000,
        .focus_address = 0x4020,
        .focus_size = bytes.size(),
        .route = Route::CpuOnly,
        .sample_size = bytes.size(),
        .diagnostic_snapshot_id = 7,
    };
    const auto record = CaptureSrtSourceReadRecord(0x4020, bytes.size(),
                                                   SrtMemoryReadFailure::None, false, bytes,
                                                   receipt);

    EXPECT_EQ(record.diagnostic_snapshot_id, 7U);
}

TEST(SrtSourceRead, CapturesGpuCompositionRoute) {
    constexpr std::array<u8, 4> bytes{0xaa, 0xbb, 0xcc, 0xdd};
    const auto record = CaptureSrtSourceReadRecord(0x8044, bytes.size(), SrtMemoryReadFailure::None,
                                                   false, bytes, GpuReceipt);

    EXPECT_EQ(record.route, Route::GpuComposed);
    EXPECT_EQ(record.snapshot_address, GpuReceipt.snapshot_address);
    EXPECT_EQ(record.snapshot_size, GpuReceipt.snapshot_size);
    EXPECT_TRUE(record.source_bytes_known);
    EXPECT_EQ(record.bytes_captured, bytes.size());
    EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(), record.first_bytes.begin()));
}

TEST(SrtSourceRead, FailedReadDoesNotPublishUntrustedBytes) {
    constexpr std::array<u8, 16> bytes{0xff};
    const auto record = CaptureSrtSourceReadRecord(
        0xaab485027cd0ULL, 4, SrtMemoryReadFailure::InvalidRequest, false, bytes, GpuReceipt);

    EXPECT_TRUE(record.request_known);
    EXPECT_EQ(record.failure, SrtMemoryReadFailure::InvalidRequest);
    EXPECT_FALSE(record.source_bytes_known);
    EXPECT_EQ(record.bytes_captured, 0U);
    EXPECT_EQ(record.route, Route::Unavailable);
    EXPECT_TRUE(std::ranges::all_of(record.first_bytes, [](u8 value) { return value == 0; }));
}

TEST(SrtSourceRead, FailedReadDoesNotClaimCompositionRoute) {
    constexpr std::array<u8, 4> bytes{0xaa, 0xbb, 0xcc, 0xdd};
    const auto record = CaptureSrtSourceReadRecord(
        0x8044, bytes.size(), SrtMemoryReadFailure::UnmappedRange, false, bytes, GpuReceipt);

    EXPECT_EQ(record.route, Route::Unavailable);
    EXPECT_EQ(record.snapshot_address, 0U);
    EXPECT_EQ(record.snapshot_size, 0U);
    EXPECT_FALSE(record.source_bytes_known);
    EXPECT_EQ(record.bytes_captured, 0U);
}

TEST(SrtSourceRead, RangeOutsideCompositionReceiptIsUnavailable) {
    constexpr std::array<u8, 2> bytes{0x12, 0x34};
    const auto record = CaptureSrtSourceReadRecord(0x9000, bytes.size(), SrtMemoryReadFailure::None,
                                                   false, bytes, GpuReceipt);

    EXPECT_EQ(record.route, Route::Unavailable);
    EXPECT_EQ(record.snapshot_address, 0U);
    EXPECT_EQ(record.snapshot_size, 0U);
    EXPECT_TRUE(record.source_bytes_known);
}

TEST(SrtSourceReadSource, PipelineLogsBoundedSourceReadRecord) {
    std::ifstream file{SHADPS4_PIPELINE_CACHE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(file.is_open());
    std::ostringstream contents;
    contents << file.rdbuf();
    const auto source = contents.str();

    const auto source_record = source.find("SrtSourceReadV1");
    ASSERT_NE(source_record, std::string::npos);
    const auto source_logger = source.rfind("const auto emit_source_read_records", source_record);
    ASSERT_NE(source_logger, std::string::npos);
    const auto target_emit = source.find("GetSrtFailureFlatDiagnostics().Emit(params.hash");
    ASSERT_NE(target_emit, std::string::npos);
    EXPECT_LT(source_logger, source_record);
    EXPECT_LT(source_record, target_emit);
    EXPECT_NE(source.substr(source_logger, source_record - source_logger).find("SourceReads()"),
              std::string::npos);
    EXPECT_NE(source.find("emit_source_read_records(row, phase)", target_emit), std::string::npos);
    EXPECT_NE(source.find("emit_success_source_reads(\"cached_success\")"), std::string::npos);
    EXPECT_NE(source.find("request_address={:#x}"), std::string::npos);
    EXPECT_NE(source.find("request_size={}"), std::string::npos);
    EXPECT_NE(source.find("route={}"), std::string::npos);
    EXPECT_NE(source.find("source_bytes_known={}"), std::string::npos);
    EXPECT_NE(source.find("diagnostic_snapshot_id={}"), std::string::npos);
    EXPECT_NE(source.find("first_bytes=[{:02x}"), std::string::npos);
}

TEST(SrtSourceReadSource, CachedBindingCursorCommitChecksMalformedResources) {
    std::ifstream file{SHADPS4_PIPELINE_CACHE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(file.is_open());
    std::ostringstream contents;
    contents << file.rdbuf();
    const auto source = contents.str();

    size_t checked_calls = 0;
    size_t search_from = 0;
    while (true) {
        const auto call = source.find("AddBindings(selected_binding)", search_from);
        if (call == std::string::npos) {
            break;
        }
        const auto guard = source.rfind("if (!selected_info->AddBindings(selected_binding))",
                                       call);
        ASSERT_NE(guard, std::string::npos);
        EXPECT_LT(guard, call);
        const auto commit = source.find("binding = selected_binding", call);
        ASSERT_NE(commit, std::string::npos);
        EXPECT_LT(call, commit);
        ++checked_calls;
        search_from = call + 1;
    }
    EXPECT_EQ(checked_calls, 2U);
}

} // namespace
} // namespace VideoCore
