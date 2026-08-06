// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/buffer_cache/fault_download.h"

namespace VideoCore {
namespace {

TEST(FaultDownload, ReservesSlotZeroForTheReportedCount) {
    const auto count = BoundFaultDownloadCount(FaultDownloadSlotCount - 1, FaultDownloadSlotCount);

    EXPECT_EQ(count.address_count, FaultDownloadSlotCount - 1);
    EXPECT_FALSE(count.overflowed);
}

TEST(FaultDownload, CapsUntrustedCountAtTheDownloadedAddressSlots) {
    const auto count = BoundFaultDownloadCount(FaultDownloadSlotCount + 37, FaultDownloadSlotCount);

    EXPECT_EQ(count.address_count, FaultDownloadSlotCount - 1);
    EXPECT_TRUE(count.overflowed);
}

TEST(FaultDownload, ReportsOverflowWhenThereAreNoAddressSlots) {
    const auto count = BoundFaultDownloadCount(1, 1);

    EXPECT_EQ(count.address_count, 0);
    EXPECT_TRUE(count.overflowed);
}

TEST(FaultDownload, KeepsAnEmptyDownloadClean) {
    const auto count = BoundFaultDownloadCount(0, FaultDownloadSlotCount);

    EXPECT_EQ(count.address_count, 0);
    EXPECT_FALSE(count.overflowed);
}

} // namespace
} // namespace VideoCore
