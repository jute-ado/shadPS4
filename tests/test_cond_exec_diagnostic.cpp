// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/cond_exec_diagnostic.h"

namespace AmdGpu {
namespace {

TEST(CondExecDiagnostic, BoundsReportsWhileRetainingSkipAndExecuteCounts) {
    CondExecDiagnosticTracker tracker{2};

    const auto first = tracker.Observe(0, true, true);
    EXPECT_TRUE(first.skip);
    EXPECT_TRUE(first.should_report);
    EXPECT_EQ(first.occurrence, 1);
    EXPECT_TRUE(first.registered);
    EXPECT_TRUE(first.gpu_dirty);
    EXPECT_EQ(first.sample_kind, CondExecSampleKind::Zero);

    const auto second = tracker.Observe(1, true, false);
    EXPECT_FALSE(second.skip);
    EXPECT_TRUE(second.should_report);
    EXPECT_EQ(second.occurrence, 2);
    EXPECT_TRUE(second.registered);
    EXPECT_FALSE(second.gpu_dirty);
    EXPECT_EQ(second.sample_kind, CondExecSampleKind::One);

    const auto third = tracker.Observe(0x100, false, false);
    EXPECT_FALSE(third.skip);
    EXPECT_FALSE(third.should_report);
    EXPECT_EQ(third.occurrence, 3);
    EXPECT_FALSE(third.registered);
    EXPECT_FALSE(third.gpu_dirty);
    EXPECT_EQ(third.sample_kind, CondExecSampleKind::OtherNonZero);

    EXPECT_EQ(tracker.Total(), 3);
    EXPECT_EQ(tracker.Skipped(), 1);
    EXPECT_EQ(tracker.Executed(), 2);
}

} // namespace
} // namespace AmdGpu
