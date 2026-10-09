// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/amdgpu/predication_policy.h"

namespace AmdGpu {
namespace {

constexpr u64 Valid = 0x8000000000000000ULL;

TEST(PredicationPolicy, ExecutesVisibleAndInvisiblePacketVariants) {
    const std::array counters{Valid | 7, Valid | 9};
    const auto visible = EvaluateZpassVisibility(counters, 1);
    ASSERT_EQ(visible, true);

    StaticPredicationState state;
    state.Set(visible, true, false);
    EXPECT_FALSE(state.ShouldSkip(true));
    state.Set(visible, false, false);
    EXPECT_TRUE(state.ShouldSkip(true));
    EXPECT_FALSE(state.ShouldSkip(false));
}

TEST(PredicationPolicy, TreatsEqualCountersAsOccluded) {
    const std::array counters{Valid | 11, Valid | 11};
    const auto visible = EvaluateZpassVisibility(counters, 1);
    ASSERT_EQ(visible, false);

    StaticPredicationState state;
    state.Set(visible, true, false);
    EXPECT_TRUE(state.ShouldSkip(true));
    EXPECT_FALSE(state.ShouldSkip(true, true));
}

TEST(PredicationPolicy, FailsOpenForIncompleteOrInvalidCounters) {
    const std::array incomplete{Valid | 1};
    const std::array invalid{1ULL, Valid | 2};

    StaticPredicationState state;
    state.Set(EvaluateZpassVisibility(incomplete, 1), true, false);
    EXPECT_FALSE(state.ShouldSkip(true));
    state.Set(EvaluateZpassVisibility(invalid, 1), true, false);
    EXPECT_FALSE(state.ShouldSkip(true));
}

TEST(PredicationPolicy, CombinesVisibilityBeforeApplyingPolarity) {
    StaticPredicationState state;
    state.Set(false, true, false);
    state.Set(true, true, true);
    EXPECT_FALSE(state.ShouldSkip(true));
    state.Set(false, false, false);
    EXPECT_FALSE(state.ShouldSkip(true));
    state.Disable();
    EXPECT_FALSE(state.ShouldSkip(true));
}

TEST(PredicationPolicy, RealSampleWindowsPreserveEmptyAndVisibleResults) {
    ZpassCounterState counter;

    const u64 empty_begin = counter.Snapshot();
    const u64 empty_end = counter.AddSamples(0);
    const std::array empty_window{empty_begin, empty_end};
    EXPECT_EQ(EvaluateZpassVisibility(empty_window, 1), false);

    const u64 visible_begin = counter.Snapshot();
    const u64 visible_end = counter.AddSamples(37);
    const std::array visible_window{visible_begin, visible_end};
    EXPECT_EQ(EvaluateZpassVisibility(visible_window, 1), true);
    EXPECT_EQ(visible_end & ~OcclusionCounterValidMask, 37);
}

} // namespace
} // namespace AmdGpu
