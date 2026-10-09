// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/buffer_cache/fault_range.h"

TEST(FaultRange, AcceptsMappedAddressSpaceRange) {
    EXPECT_TRUE(VideoCore::IsCacheableFaultRange(0x4000, 0x8000, 1ULL << 40));
}

TEST(FaultRange, RejectsNullPage) {
    EXPECT_FALSE(VideoCore::IsCacheableFaultRange(0, 0x4000, 1ULL << 40));
}

TEST(FaultRange, RejectsRangeOutsideAddressSpace) {
    EXPECT_FALSE(VideoCore::IsCacheableFaultRange(1ULL << 40, (1ULL << 40) + 0x4000, 1ULL << 40));
}

TEST(FaultRange, RejectsEmptyOrReversedRange) {
    EXPECT_FALSE(VideoCore::IsCacheableFaultRange(0x4000, 0x4000, 1ULL << 40));
    EXPECT_FALSE(VideoCore::IsCacheableFaultRange(0x8000, 0x4000, 1ULL << 40));
}

TEST(FaultRange, RejectsRangeTooLargeForBufferCache) {
    EXPECT_FALSE(VideoCore::IsCacheableFaultRange(0x4000, 0x4000 + (1ULL << 32), 1ULL << 40));
}

TEST(FaultRange, BoundsGpuReportedCountToStoredEntries) {
    EXPECT_EQ(VideoCore::BoundRecordedFaultCount(0, 1023), 0);
    EXPECT_EQ(VideoCore::BoundRecordedFaultCount(1023, 1023), 1023);
    EXPECT_EQ(VideoCore::BoundRecordedFaultCount(1024, 1023), 1023);
    EXPECT_EQ(VideoCore::BoundRecordedFaultCount(1'000'000, 1023), 1023);
}
