// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/fault_buffer_layout.h"

namespace VideoCore {
namespace {

TEST(FaultBufferCount, PreservesCountsThatFitTheDownloadArea) {
    EXPECT_EQ(BoundFaultCount(0), 0);
    EXPECT_EQ(BoundFaultCount(MaxStoredPageFaults), MaxStoredPageFaults);
}

TEST(FaultBufferCount, ReservesTheFirstDownloadEntryForTheCount) {
    EXPECT_EQ(BoundFaultCount(FaultDownloadEntries), MaxStoredPageFaults);
}

TEST(FaultBufferCount, ClampsArbitrarilyLargeAtomicCounts) {
    EXPECT_EQ(BoundFaultCount(std::numeric_limits<u64>::max()), MaxStoredPageFaults);
}

} // namespace
} // namespace VideoCore
