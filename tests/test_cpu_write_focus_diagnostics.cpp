// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <gtest/gtest.h>

#include "video_core/buffer_cache/cpu_page_write_snapshot.h"

namespace {

std::string ReadFocusSource(const std::filesystem::path& path) {
    std::ifstream source{path, std::ios::binary};
    EXPECT_TRUE(source.is_open());
    std::string text{std::istreambuf_iterator<char>{source}, std::istreambuf_iterator<char>{}};
    std::erase(text, '\r');
    return text;
}

TEST(CpuWriteFocusDiagnosticsWiring, CompositionRetainsFocusedCpuAndOwnerMetadata) {
    const auto source = ReadFocusSource(SHADPS4_INVALIDATION_BUFFER_SOURCE_PATH);
    EXPECT_NE(source.find("cpu_write_focus"), std::string::npos);
    EXPECT_NE(source.find("focus_owners"), std::string::npos);
}

TEST(CpuWriteFocusDiagnosticsWiring, TargetFailureEmitsBoundedReceiptMetadata) {
    const auto source_path =
        std::filesystem::path{SHADPS4_INVALIDATION_BUFFER_SOURCE_PATH}.parent_path().parent_path() /
        "renderer_vulkan/vk_pipeline_cache.cpp";
    const auto source = ReadFocusSource(source_path);
    const auto failure = source.find("GetSrtFailureFlatDiagnostics().EmitFailure(");
    const auto cpu = source.find("SrtCpuWriteFocusV1", failure);
    const auto owner = source.find("SrtFocusOwnerV1", failure);
    const auto success = source.find("const auto emit_success_source_reads", failure);
    ASSERT_NE(cpu, std::string::npos);
    ASSERT_NE(owner, std::string::npos);
    ASSERT_NE(success, std::string::npos);
    EXPECT_LT(cpu, success);
    EXPECT_LT(owner, success);
}

using FocusPage = std::array<u8, VideoCore::TRACKER_BYTES_PER_PAGE>;
constexpr VAddr FocusPageAddress = 0x4000;
constexpr size_t FocusOffset = 0x70;

void StoreFocusWord(FocusPage& page, size_t offset, u32 value) {
    std::memcpy(page.data() + offset, &value, sizeof(value));
}

TEST(CpuWriteFocusDiagnostics, MissingBeforeImageIsExplicitAndCannotChangeOwnership) {
    VideoCore::CpuPageWriteTracker tracker;
    FocusPage cpu{};
    StoreFocusWord(cpu, FocusOffset, 0x1234);
    FocusPage output{};
    output.fill(0x77);
    VideoCore::CpuWriteFocusDiagnostic diagnostic;
    ASSERT_TRUE(tracker.ApplyPendingWrites(FocusPageAddress, cpu, FocusPageAddress, output,
                                           &diagnostic, FocusPageAddress + FocusOffset, 8));
    EXPECT_TRUE(diagnostic.available);
    EXPECT_EQ(diagnostic.word_count, 2u);
    EXPECT_EQ(diagnostic.snapshot_mask, 0u);
    EXPECT_EQ(diagnostic.forced_word_mask, 0u);
    EXPECT_EQ(diagnostic.changed_word_mask, 0u);
    EXPECT_EQ(diagnostic.captured[0], 0x1234u);
    EXPECT_EQ(output[FocusOffset], 0x77);
}

TEST(CpuWriteFocusDiagnostics, ForcedAndChangedWordsRemainDistinctObservations) {
    VideoCore::CpuPageWriteTracker tracker;
    FocusPage cpu{};
    StoreFocusWord(cpu, FocusOffset, 0x1000);
    StoreFocusWord(cpu, FocusOffset + 4, 2);
    ASSERT_TRUE(tracker.Capture(FocusPageAddress, cpu, FocusOffset + 4, 4));
    StoreFocusWord(cpu, FocusOffset, 0x2000);
    FocusPage output{};
    VideoCore::CpuWriteFocusDiagnostic diagnostic;
    ASSERT_TRUE(tracker.ApplyPendingWrites(FocusPageAddress, cpu, FocusPageAddress, output,
                                           &diagnostic, FocusPageAddress + FocusOffset, 8));
    EXPECT_EQ(diagnostic.snapshot_mask, 3u);
    EXPECT_EQ(diagnostic.forced_word_mask, 2u);
    EXPECT_EQ(diagnostic.changed_word_mask, 1u);
    EXPECT_EQ(diagnostic.mapping_overlap_mask, 0u);
    EXPECT_EQ(diagnostic.before[0], 0x1000u);
    EXPECT_EQ(diagnostic.before[1], 2u);
    EXPECT_EQ(diagnostic.captured[0], 0x2000u);
    EXPECT_TRUE(std::equal(cpu.begin() + FocusOffset, cpu.begin() + FocusOffset + 8,
                           output.begin() + FocusOffset));
}

TEST(CpuWriteFocusDiagnostics, UnrecordedSameValueStoreIsNotMisreportedAsKnownIntent) {
    VideoCore::CpuPageWriteTracker tracker;
    FocusPage cpu{};
    StoreFocusWord(cpu, FocusOffset, 0x1000);
    ASSERT_TRUE(tracker.Capture(FocusPageAddress, cpu, 0x20, 8));
    StoreFocusWord(cpu, FocusOffset, 0x1000);
    FocusPage output{};
    output.fill(0x77);
    VideoCore::CpuWriteFocusDiagnostic diagnostic;
    ASSERT_TRUE(tracker.ApplyPendingWrites(FocusPageAddress, cpu, FocusPageAddress, output,
                                           &diagnostic, FocusPageAddress + FocusOffset, 8));
    EXPECT_EQ(diagnostic.snapshot_mask, 3u);
    EXPECT_EQ(diagnostic.forced_word_mask, 0u);
    EXPECT_EQ(diagnostic.changed_word_mask, 0u);
    EXPECT_EQ(output[FocusOffset], 0x77); // Known limitation; diagnostics must not mask it.
}

TEST(CpuWriteFocusDiagnostics, CrossPageFocusSeparatesSnapshotFromMappingMarker) {
    VideoCore::CpuPageWriteTracker tracker;
    FocusPage page{};
    ASSERT_TRUE(tracker.Capture(FocusPageAddress, page, page.size() - 4, 4));
    tracker.ForceMappingUpload(FocusPageAddress + page.size(), 4);
    const VAddr capture_address = FocusPageAddress + page.size() - 4;
    std::array<u8, 8> cpu{};
    std::array<u8, 8> output{};
    output.fill(0x77);
    VideoCore::CpuWriteFocusDiagnostic diagnostic;
    ASSERT_TRUE(tracker.ApplyPendingWrites(capture_address, cpu, capture_address, output,
                                           &diagnostic, capture_address, 8));
    EXPECT_EQ(diagnostic.word_count, 2u);
    EXPECT_EQ(diagnostic.snapshot_mask, 1u);
    EXPECT_EQ(diagnostic.forced_word_mask, 1u);
    EXPECT_EQ(diagnostic.mapping_overlap_mask, 2u);
    EXPECT_EQ(output, cpu);
}

TEST(CpuWriteFocusDiagnostics, FocusIsBoundedToEightWholeWords) {
    VideoCore::CpuPageWriteTracker tracker;
    FocusPage cpu{};
    FocusPage output{};
    VideoCore::CpuWriteFocusDiagnostic diagnostic;
    ASSERT_TRUE(tracker.ApplyPendingWrites(FocusPageAddress, cpu, FocusPageAddress, output,
                                           &diagnostic, FocusPageAddress + FocusOffset + 3, 64));
    EXPECT_TRUE(diagnostic.available);
    EXPECT_EQ(diagnostic.word_address, FocusPageAddress + FocusOffset);
    EXPECT_EQ(diagnostic.word_count, 8u);
}

TEST(CpuWriteFocusDiagnostics, InvalidFocusIsUnavailableWithoutChangingReconciliation) {
    VideoCore::CpuPageWriteTracker tracker;
    FocusPage cpu{};
    FocusPage output{};
    VideoCore::CpuWriteFocusDiagnostic diagnostic;
    for (const auto focus : {FocusPageAddress - 1, FocusPageAddress + cpu.size()}) {
        ASSERT_TRUE(tracker.ApplyPendingWrites(FocusPageAddress, cpu, FocusPageAddress, output,
                                               &diagnostic, focus, 8));
        EXPECT_FALSE(diagnostic.available);
        EXPECT_EQ(diagnostic.word_count, 0u);
    }
    diagnostic.available = true;
    EXPECT_FALSE(tracker.ApplyPendingWrites(FocusPageAddress + 1, cpu, FocusPageAddress + 1, output,
                                            &diagnostic, FocusPageAddress + 1, 8));
    EXPECT_FALSE(diagnostic.available);
}

TEST(CpuWriteFocusDiagnostics, DiagnosticDoesNotConsumeOrChangeRepeatedOverlay) {
    VideoCore::CpuPageWriteTracker tracker;
    FocusPage cpu{};
    ASSERT_TRUE(tracker.Capture(FocusPageAddress, cpu, FocusOffset, 8));
    FocusPage plain{};
    FocusPage observed{};
    plain.fill(0x77);
    observed = plain;
    ASSERT_TRUE(tracker.ApplyPendingWrites(FocusPageAddress, cpu, FocusPageAddress, plain));
    VideoCore::CpuWriteFocusDiagnostic diagnostic;
    ASSERT_TRUE(tracker.ApplyPendingWrites(FocusPageAddress, cpu, FocusPageAddress, observed,
                                           &diagnostic, FocusPageAddress + FocusOffset, 8));
    EXPECT_EQ(observed, plain);
    unsigned ranges = 0;
    EXPECT_TRUE(tracker.Consume(FocusPageAddress, cpu, 0, cpu.size(), [&](auto range) {
        ++ranges;
        EXPECT_EQ(range.offset, FocusOffset);
        EXPECT_EQ(range.size, 8u);
    }));
    EXPECT_EQ(ranges, 1u);
}

TEST(CpuWriteFocusDiagnostics, BeforeWordAccessorRejectsOutOfRange) {
    FocusPage cpu{};
    VideoCore::CpuPageWriteSnapshot snapshot{cpu, 0, 4};
    EXPECT_TRUE(snapshot.DiagnosticBeforeWord(0));
    EXPECT_FALSE(snapshot.DiagnosticBeforeWord(snapshot.WordsPerPage));
    EXPECT_FALSE(snapshot.DiagnosticForcedWord(snapshot.WordsPerPage));
}

} // namespace
