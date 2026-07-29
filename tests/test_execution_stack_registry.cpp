// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>

#include "core/execution_stack_registry.h"

TEST(ExecutionStackRegistry, ExcludesRegisteredStacksWithoutDroppingAdjacentPages) {
    Core::ExecutionStackRegistry registry;

    ASSERT_TRUE(registry.Register(0x2000, 0x2000));
    EXPECT_FALSE(registry.Overlaps(0x1000, 0x1000));
    EXPECT_TRUE(registry.Overlaps(0x1fff, 2));
    EXPECT_TRUE(registry.Overlaps(0x3000, 0x1000));
    EXPECT_FALSE(registry.Overlaps(0x4000, 0x1000));

    const std::vector<Core::ExecutionStackRange> expected{
        {0x1000, 0x1000},
        {0x4000, 0x2000},
    };
    EXPECT_EQ(registry.GetWatchableRanges(0x1000, 0x5000), expected);
}

TEST(ExecutionStackRegistry, ReferenceCountsExactRegistrationsAndMergesOverlaps) {
    Core::ExecutionStackRegistry registry;

    ASSERT_TRUE(registry.Register(0x2000, 0x2000));
    ASSERT_TRUE(registry.Register(0x2000, 0x2000));
    ASSERT_TRUE(registry.Register(0x3000, 0x3000));
    EXPECT_TRUE(registry.Overlaps(0x5000, 1));
    EXPECT_TRUE(registry.GetWatchableRanges(0x1000, 0x6000) ==
                (std::vector<Core::ExecutionStackRange>{{0x1000, 0x1000}, {0x6000, 0x1000}}));

    EXPECT_TRUE(registry.Unregister(0x2000, 0x2000));
    EXPECT_TRUE(registry.Overlaps(0x2000, 1));
    EXPECT_TRUE(registry.Unregister(0x2000, 0x2000));
    EXPECT_FALSE(registry.Overlaps(0x2000, 0x1000));
    EXPECT_TRUE(registry.Overlaps(0x3000, 0x3000));
    EXPECT_FALSE(registry.Unregister(0x2000, 0x2000));

    EXPECT_TRUE(registry.Unregister(0x3000, 0x3000));
    EXPECT_FALSE(registry.Overlaps(0x1000, 0x6000));
}

TEST(ExecutionStackRegistry, ExcludesWholePagesForUnalignedStacks) {
    Core::ExecutionStackRegistry registry;

    ASSERT_TRUE(registry.Register(0x2800, 0x100));
    EXPECT_TRUE(registry.Overlaps(0x2000, 1));
    EXPECT_TRUE(registry.Overlaps(0x2fff, 1));
    EXPECT_FALSE(registry.Overlaps(0x3000, 1));
    EXPECT_TRUE(registry.GetWatchableRanges(0x1000, 0x4000) ==
                (std::vector<Core::ExecutionStackRange>{{0x1000, 0x1000}, {0x3000, 0x2000}}));
    EXPECT_TRUE(registry.Unregister(0x2800, 0x100));
}

TEST(ExecutionStackRegistry, RejectsEmptyAndOverflowingRanges) {
    Core::ExecutionStackRegistry registry;
    constexpr VAddr max = std::numeric_limits<VAddr>::max();

    EXPECT_FALSE(registry.Register(0x1000, 0));
    EXPECT_FALSE(registry.Register(max - 1, 2));
    EXPECT_FALSE(registry.Unregister(0x1000, 0));
    EXPECT_FALSE(registry.Overlaps(max - 1, 2));
    EXPECT_TRUE(registry.GetWatchableRanges(0x1000, 0).empty());
    EXPECT_TRUE(registry.GetWatchableRanges(max - 1, 2).empty());
}
