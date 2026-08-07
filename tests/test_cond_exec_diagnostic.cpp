// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/cond_exec_diagnostic.h"

namespace AmdGpu {
namespace {

TEST(CondExecDiagnostic, BoundsReportsWhileRetainingSkipAndExecuteCounts) {
    CondExecDiagnosticTracker tracker{2};

    const auto first = tracker.Observe(false);
    EXPECT_TRUE(first.skip);
    EXPECT_TRUE(first.should_report);
    EXPECT_EQ(first.occurrence, 1);

    const auto second = tracker.Observe(true);
    EXPECT_FALSE(second.skip);
    EXPECT_TRUE(second.should_report);
    EXPECT_EQ(second.occurrence, 2);

    const auto third = tracker.Observe(false);
    EXPECT_TRUE(third.skip);
    EXPECT_FALSE(third.should_report);
    EXPECT_EQ(third.occurrence, 3);

    EXPECT_EQ(tracker.Total(), 3);
    EXPECT_EQ(tracker.Skipped(), 2);
    EXPECT_EQ(tracker.Executed(), 1);
}

} // namespace
} // namespace AmdGpu
