// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/ipc/allocation_callsite_summary.h"

TEST(AllocationCallsiteSummary, KeepsSizeSeparatedPerCallerAndRanksByTotalBytes) {
    Core::Ipc::AllocationCallsiteSummary summary;
    summary.Insert(0x1000, 64);
    summary.Insert(0x1000, 64);
    summary.Insert(0x1000, 128);
    summary.Insert(0x2000, 1024);

    const auto top = summary.TopByTotalBytes();
    ASSERT_EQ(top.size(), 3);
    EXPECT_EQ(top[0].caller, 0x2000);
    EXPECT_EQ(top[0].total_bytes, 1024);
    EXPECT_EQ(top[1].caller, 0x1000);
    EXPECT_EQ(top[1].size, 64);
    EXPECT_EQ(top[1].count, 2);
    EXPECT_EQ(top[2].size, 128);
}

TEST(SampledAllocationTracker, ReportsOnlyAllocationsStillLiveAfterFreesAndCollisions) {
    Core::Ipc::SampledAllocationTracker<8> tracker;
    auto* first = reinterpret_cast<void*>(0x1000);
    auto* colliding = reinterpret_cast<void*>(0x1080);
    auto* replacement = reinterpret_cast<void*>(0x2000);

    ASSERT_TRUE(tracker.Add(first, 0xaaaa, 32));
    ASSERT_TRUE(tracker.Add(colliding, 0xbbbb, 64));
    ASSERT_TRUE(tracker.Remove(first));
    ASSERT_TRUE(tracker.Add(replacement, 0xcccc, 128));
    EXPECT_FALSE(tracker.Remove(reinterpret_cast<void*>(0x3000)));

    const auto top = tracker.Summarize().TopByTotalBytes();
    ASSERT_EQ(top.size(), 2);
    EXPECT_EQ(top[0].caller, 0xcccc);
    EXPECT_EQ(top[1].caller, 0xbbbb);
}
