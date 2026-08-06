// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/buffer_cache/fault_download.h"

namespace VideoCore {
namespace {

TEST(FaultDownload, ReservesSlotZeroForTheReportedCount) {
    constexpr size_t DownloadSlotCount = 1024;

    const auto count = BoundFaultDownloadCount(DownloadSlotCount - 1, DownloadSlotCount);

    EXPECT_EQ(count.address_count, DownloadSlotCount - 1);
    EXPECT_FALSE(count.overflowed);
}

TEST(FaultDownload, CapsUntrustedCountAtTheDownloadedAddressSlots) {
    constexpr size_t DownloadSlotCount = 1024;

    const auto count = BoundFaultDownloadCount(DownloadSlotCount + 37, DownloadSlotCount);

    EXPECT_EQ(count.address_count, DownloadSlotCount - 1);
    EXPECT_TRUE(count.overflowed);
}

TEST(FaultDownload, ReportsOverflowWhenThereAreNoAddressSlots) {
    const auto count = BoundFaultDownloadCount(1, 1);

    EXPECT_EQ(count.address_count, 0);
    EXPECT_TRUE(count.overflowed);
}

TEST(FaultDownload, KeepsAnEmptyDownloadClean) {
    const auto count = BoundFaultDownloadCount(0, 1024);

    EXPECT_EQ(count.address_count, 0);
    EXPECT_FALSE(count.overflowed);
}

} // namespace
} // namespace VideoCore
