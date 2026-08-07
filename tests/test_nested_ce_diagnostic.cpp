// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/nested_ce_diagnostic.h"

namespace AmdGpu {
namespace {

TEST(NestedCeDiagnostic, BoundsHazardReportsWhileRetainingEntryCounts) {
    NestedCeDiagnosticTracker tracker{2};

    tracker.ObserveTopLevel(false);
    tracker.ObserveTopLevel(true);

    const auto first = tracker.ObserveNested(0, 0, false);
    EXPECT_FALSE(first.hazard);
    EXPECT_TRUE(first.should_report);
    EXPECT_EQ(first.occurrence, 1);

    const auto second = tracker.ObserveNested(2, 1, true);
    EXPECT_TRUE(second.hazard);
    EXPECT_TRUE(second.should_report);
    EXPECT_EQ(second.occurrence, 2);
    EXPECT_EQ(second.ce_count, 2);
    EXPECT_EQ(second.de_count, 1);
    EXPECT_EQ(second.difference, 1);
    EXPECT_TRUE(second.parent_ce_unfinished);

    const auto third = tracker.ObserveNested(0, 1, false);
    EXPECT_TRUE(third.hazard);
    EXPECT_FALSE(third.should_report);
    EXPECT_EQ(third.occurrence, 3);

    EXPECT_EQ(tracker.TopLevelEntries(), 2);
    EXPECT_EQ(tracker.TopLevelEntriesWithCcb(), 1);
    EXPECT_EQ(tracker.NestedEntries(), 3);
    EXPECT_EQ(tracker.NestedHazards(), 2);
}

} // namespace
} // namespace AmdGpu
