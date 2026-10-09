// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <vector>
#include <gtest/gtest.h>

#include "video_core/buffer_cache/cpu_page_write_snapshot.h"

namespace {

using VideoCore::CpuPageUploadRange;
using VideoCore::CpuPageWriteSnapshot;
using VideoCore::CpuPageWriteTracker;
using VideoCore::TRACKER_BYTES_PER_PAGE;

std::vector<CpuPageUploadRange> UploadRanges(const CpuPageWriteSnapshot& snapshot,
                                             const std::array<u8, TRACKER_BYTES_PER_PAGE>& current,
                                             size_t offset = 0,
                                             size_t size = TRACKER_BYTES_PER_PAGE) {
    std::vector<CpuPageUploadRange> ranges;
    snapshot.ForEachUploadRange(current, offset, size,
                                [&](CpuPageUploadRange range) { ranges.push_back(range); });
    return ranges;
}

} // namespace

TEST(CpuPageWriteSnapshot, UploadsOnlyCpuWordsChangedAfterGpuOwnership) {
    std::array<u8, TRACKER_BYTES_PER_PAGE> before{};
    auto current = before;
    current[0x120] = 0x11;
    current[0x127] = 0x77;

    const CpuPageWriteSnapshot snapshot{before, 0x120, 8};

    EXPECT_EQ(UploadRanges(snapshot, current),
              (std::vector{CpuPageUploadRange{.offset = 0x120, .size = 8}}));
}

TEST(CpuPageWriteSnapshot, PreservesSameValueWritesAtTheFaultingWord) {
    std::array<u8, TRACKER_BYTES_PER_PAGE> page{};
    const CpuPageWriteSnapshot snapshot{page, 0x943, 1};

    EXPECT_EQ(UploadRanges(snapshot, page),
              (std::vector{CpuPageUploadRange{.offset = 0x940, .size = 4}}));
}

TEST(CpuPageWriteSnapshot, CoalescesAdjacentChangedWords) {
    std::array<u8, TRACKER_BYTES_PER_PAGE> before{};
    auto current = before;
    current[0x100] = 1;
    current[0x104] = 2;
    const CpuPageWriteSnapshot snapshot{before, 0x100, 1};

    EXPECT_EQ(UploadRanges(snapshot, current),
              (std::vector{CpuPageUploadRange{.offset = 0x100, .size = 8}}));
}

TEST(CpuPageWriteSnapshot, KeepsSeparatedChangedWordsAsSeparateUploads) {
    std::array<u8, TRACKER_BYTES_PER_PAGE> before{};
    auto current = before;
    current[0x100] = 1;
    current[0x10c] = 2;
    const CpuPageWriteSnapshot snapshot{before, 0x100, 1};

    EXPECT_EQ(UploadRanges(snapshot, current), (std::vector{
                                                   CpuPageUploadRange{.offset = 0x100, .size = 4},
                                                   CpuPageUploadRange{.offset = 0x10c, .size = 4},
                                               }));
}

TEST(CpuPageWriteSnapshot, LimitsUploadsToTheRequestedPageInterval) {
    std::array<u8, TRACKER_BYTES_PER_PAGE> before{};
    auto current = before;
    current[0x100] = 1;
    current[0x208] = 2;
    current[0x300] = 3;
    const CpuPageWriteSnapshot snapshot{before, 0x208, 1};

    EXPECT_EQ(UploadRanges(snapshot, current, 0x200, 0x20),
              (std::vector{CpuPageUploadRange{.offset = 0x208, .size = 4}}));
}

TEST(CpuPageWriteSnapshot, AccumulatesFaultingWordsBeforeThePageIsConsumed) {
    std::array<u8, TRACKER_BYTES_PER_PAGE> page{};
    CpuPageWriteSnapshot snapshot{page, 0x100, 1};
    snapshot.MarkWrite(0x200, 8);

    EXPECT_EQ(UploadRanges(snapshot, page), (std::vector{
                                                CpuPageUploadRange{.offset = 0x100, .size = 4},
                                                CpuPageUploadRange{.offset = 0x200, .size = 8},
                                            }));
}

TEST(CpuPageWriteTracker, ConsumesCapturedPagesOnlyOnce) {
    std::array<u8, TRACKER_BYTES_PER_PAGE> before{};
    auto current = before;
    current[0x120] = 1;
    CpuPageWriteTracker tracker;
    ASSERT_TRUE(tracker.Capture(0x4000, before, 0x120, 1));

    std::vector<CpuPageUploadRange> ranges;
    EXPECT_TRUE(tracker.Consume(0x4000, current, 0, TRACKER_BYTES_PER_PAGE,
                                [&](CpuPageUploadRange range) { ranges.push_back(range); }));
    EXPECT_EQ(ranges, (std::vector{CpuPageUploadRange{.offset = 0x120, .size = 4}}));
    EXPECT_FALSE(
        tracker.Consume(0x4000, current, 0, TRACKER_BYTES_PER_PAGE, [](CpuPageUploadRange) {}));
}

TEST(CpuPageWriteTracker, KeepsTheOriginalPageAcrossRepeatedFaultRecords) {
    std::array<u8, TRACKER_BYTES_PER_PAGE> before{};
    auto after_first_write = before;
    after_first_write[0x100] = 1;
    auto current = after_first_write;
    current[0x200] = 2;
    CpuPageWriteTracker tracker;
    ASSERT_TRUE(tracker.Capture(0x8000, before, 0x100, 1));
    ASSERT_TRUE(tracker.Capture(0x8000, after_first_write, 0x200, 1));

    std::vector<CpuPageUploadRange> ranges;
    ASSERT_TRUE(tracker.Consume(0x8000, current, 0, TRACKER_BYTES_PER_PAGE,
                                [&](CpuPageUploadRange range) { ranges.push_back(range); }));
    EXPECT_EQ(ranges, (std::vector{
                          CpuPageUploadRange{.offset = 0x100, .size = 4},
                          CpuPageUploadRange{.offset = 0x200, .size = 4},
                      }));
}

TEST(CpuPageWriteTracker, DiscardsSnapshotsAcrossAnAddressRange) {
    std::array<u8, TRACKER_BYTES_PER_PAGE> page{};
    CpuPageWriteTracker tracker;
    ASSERT_TRUE(tracker.Capture(0x4000, page, 0x100, 1));
    ASSERT_TRUE(tracker.Capture(0x5000, page, 0x100, 1));
    ASSERT_TRUE(tracker.Capture(0x6000, page, 0x100, 1));

    tracker.Discard(0x4ff0, 0x1010);

    EXPECT_FALSE(
        tracker.Consume(0x4000, page, 0, TRACKER_BYTES_PER_PAGE, [](CpuPageUploadRange) {}));
    EXPECT_FALSE(
        tracker.Consume(0x5000, page, 0, TRACKER_BYTES_PER_PAGE, [](CpuPageUploadRange) {}));
    EXPECT_TRUE(
        tracker.Consume(0x6000, page, 0, TRACKER_BYTES_PER_PAGE, [](CpuPageUploadRange) {}));
}

TEST(CpuPageWriteTracker, PrivateOverlayIsNonconsumingAndPreservesGpuNeighbours) {
    constexpr VAddr Page = 0x4000;
    std::array<u8, TRACKER_BYTES_PER_PAGE> before{};
    before.fill(0x11);
    auto captured = before;
    captured[0x200] = 0x22;
    captured[0x208] = 0x33;
    CpuPageWriteTracker tracker;
    ASSERT_TRUE(tracker.Capture(Page, before, 0x100, 1));

    for (int repetition = 0; repetition < 2; ++repetition) {
        std::array<u8, TRACKER_BYTES_PER_PAGE> output{};
        output.fill(0x77);
        ASSERT_TRUE(tracker.ApplyPendingWrites(Page, captured, Page, output));
        EXPECT_EQ(output[0x100], 0x11); // Same-value faulting word still belongs to CPU.
        EXPECT_EQ(output[0x103], 0x11);
        EXPECT_EQ(output[0x104], 0x77);
        EXPECT_EQ(output[0x200], 0x22);
        EXPECT_EQ(output[0x204], 0x77);
        EXPECT_EQ(output[0x208], 0x33);
        EXPECT_EQ(output[0x20c], 0x77);
    }
    std::vector<CpuPageUploadRange> pending;
    ASSERT_TRUE(tracker.Consume(Page, captured, 0, captured.size(),
                                [&](auto range) { pending.push_back(range); }));
    EXPECT_EQ(pending, (std::vector{CpuPageUploadRange{0x100, 4}, CpuPageUploadRange{0x200, 4},
                                    CpuPageUploadRange{0x208, 4}}));
}

TEST(CpuPageWriteTracker, PrivateOverlayClipsPartialWordsAcrossPageBoundary) {
    constexpr VAddr Page = 0x4000;
    std::array<u8, TRACKER_BYTES_PER_PAGE> before{};
    CpuPageWriteTracker tracker;
    ASSERT_TRUE(tracker.Capture(Page, before, TRACKER_BYTES_PER_PAGE - 1, 1));
    ASSERT_TRUE(tracker.Capture(Page + TRACKER_BYTES_PER_PAGE, before, 0, 1));
    const VAddr capture_address = Page + TRACKER_BYTES_PER_PAGE - 4;
    const std::array<u8, 8> captured{1, 2, 3, 4, 5, 6, 7, 8};
    std::array<u8, 8> output{};
    output.fill(0x77);
    ASSERT_TRUE(tracker.ApplyPendingWrites(capture_address, captured, capture_address + 1,
                                           std::span{output}.subspan(1, 6)));
    EXPECT_EQ(output, (std::array<u8, 8>{0x77, 2, 3, 4, 5, 6, 7, 0x77}));
    EXPECT_TRUE(tracker.Consume(Page, before, 0, before.size(), [](auto) {}));
    EXPECT_TRUE(
        tracker.Consume(Page + TRACKER_BYTES_PER_PAGE, before, 0, before.size(), [](auto) {}));
}

TEST(CpuPageWriteTracker, PartialOutputUsesWholeCapturedWordForDirtyDecision) {
    constexpr VAddr Page = 0x4000;
    std::array<u8, TRACKER_BYTES_PER_PAGE> before{};
    CpuPageWriteTracker tracker;
    ASSERT_TRUE(tracker.Capture(Page, before, 0x200, 1));
    // Only a byte outside the requested output changed; the eventual upload still owns
    // the whole word, including the requested first byte which remained zero.
    const std::array<u8, 4> captured{0, 0, 1, 0};
    std::array<u8, 1> output{0x77};
    ASSERT_TRUE(tracker.ApplyPendingWrites(Page + 0x100, captured, Page + 0x100, output));
    EXPECT_EQ(output[0], 0);
}

TEST(CpuPageWriteTracker, PrivateOverlayWithoutBeforeImageDoesNotReplaceGpuBytes) {
    CpuPageWriteTracker tracker;
    const std::array<u8, 4> captured{1, 2, 3, 4};
    std::array<u8, 4> output{5, 6, 7, 8};
    ASSERT_TRUE(tracker.ApplyPendingWrites(0x4000, captured, 0x4000, output));
    EXPECT_EQ(output, (std::array<u8, 4>{5, 6, 7, 8}));
}

TEST(CpuPageWriteTracker, InvalidPrivateCaptureLeavesOutputUnchanged) {
    CpuPageWriteTracker tracker;
    const std::array<u8, 8> captured{};
    std::array<u8, 4> output{1, 2, 3, 4};
    EXPECT_FALSE(tracker.ApplyPendingWrites(0x4001, captured, 0x4001, output));
    EXPECT_FALSE(tracker.ApplyPendingWrites(0x4000, std::span{captured}.first(7), 0x4000, output));
    EXPECT_FALSE(tracker.ApplyPendingWrites(0x4000, captured, 0x3fff, output));
    EXPECT_FALSE(tracker.ApplyPendingWrites(0x4000, captured, 0x4006, output));
    constexpr VAddr LastAlignedWord = std::numeric_limits<VAddr>::max() - 3;
    EXPECT_FALSE(tracker.ApplyPendingWrites(LastAlignedWord, captured, LastAlignedWord, output));
    EXPECT_EQ(output, (std::array<u8, 4>{1, 2, 3, 4}));
}

TEST(CpuPageWriteSnapshot, PlansMinimalWholeWordCaptureAndRejectsOverflow) {
    using VideoCore::PlanCpuSnapshotCapture;
    const auto clipped = PlanCpuSnapshotCapture(0x4fff, 2);
    ASSERT_TRUE(clipped);
    EXPECT_EQ(clipped->address, 0x4ffc);
    EXPECT_EQ(clipped->size, 8);
    const auto aligned = PlanCpuSnapshotCapture(0x4000, 16);
    ASSERT_TRUE(aligned);
    EXPECT_EQ(aligned->address, 0x4000);
    EXPECT_EQ(aligned->size, 16);
    EXPECT_FALSE(PlanCpuSnapshotCapture(0x4000, 0));
    EXPECT_FALSE(PlanCpuSnapshotCapture(std::numeric_limits<VAddr>::max() - 1, 8));
    EXPECT_FALSE(PlanCpuSnapshotCapture(std::numeric_limits<VAddr>::max() - 4, 4));
}
