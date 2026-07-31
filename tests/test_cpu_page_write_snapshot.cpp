// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <vector>
#include <gtest/gtest.h>

#include "video_core/buffer_cache/cpu_page_write_snapshot.h"

namespace {

using VideoCore::CpuPageUploadRange;
using VideoCore::CpuPageWriteSnapshot;
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

    EXPECT_EQ(UploadRanges(snapshot, current),
              (std::vector{
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

    EXPECT_EQ(UploadRanges(snapshot, page),
              (std::vector{
                  CpuPageUploadRange{.offset = 0x100, .size = 4},
                  CpuPageUploadRange{.offset = 0x200, .size = 8},
              }));
}
