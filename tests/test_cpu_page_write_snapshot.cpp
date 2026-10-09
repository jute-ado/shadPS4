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
