// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdint>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/buffer_access_interval_provenance.h"

namespace {

using RangeState = VideoCore::BasicBufferAccessRangeState<std::uint32_t>;

TEST(BufferAccessRangeState, DisjointTransitionsRetainTheirIndependentPriorAccess) {
    RangeState state{128, 0x01, 0x10};

    const auto shader = state.Transition(0, 64, 0x06, 0x20);
    const auto geometry = state.Transition(64, 64, 0x08, 0x40);

    EXPECT_TRUE(shader.requires_barrier);
    EXPECT_EQ(shader.prior_access, 0x01);
    EXPECT_EQ(shader.prior_stages, 0x10);
    EXPECT_TRUE(geometry.requires_barrier);
    EXPECT_EQ(geometry.prior_access, 0x01);
    EXPECT_EQ(geometry.prior_stages, 0x10);
    EXPECT_EQ(state.IntervalCount(), 2);
}

TEST(BufferAccessRangeState, RepeatedMatchingAccessNeedsNoBarrierOrStateSplit) {
    RangeState state{128, 0x01, 0x10};

    const auto repeated = state.Transition(32, 64, 0x01, 0x10);

    EXPECT_FALSE(repeated.requires_barrier);
    EXPECT_EQ(repeated.prior_access, 0x01);
    EXPECT_EQ(repeated.prior_stages, 0x10);
    EXPECT_EQ(state.IntervalCount(), 1);
}

TEST(BufferAccessRangeState, OverlapUnionsOnlyTheCoveredPriorIntervals) {
    RangeState state{192, 0x01, 0x10};
    const auto first = state.Transition(0, 64, 0x02, 0x20);
    const auto last = state.Transition(128, 64, 0x04, 0x40);
    EXPECT_TRUE(first.requires_barrier);
    EXPECT_TRUE(last.requires_barrier);

    const auto overlap = state.Transition(32, 128, 0x08, 0x80);

    EXPECT_TRUE(overlap.requires_barrier);
    EXPECT_EQ(overlap.prior_access, 0x01 | 0x02 | 0x04);
    EXPECT_EQ(overlap.prior_stages, 0x10 | 0x20 | 0x40);
}

TEST(BufferAccessRangeState, CapacityCoarseningNeverDiscardsPriorDependencies) {
    VideoCore::BasicBufferAccessRangeState<std::uint32_t, std::uint32_t, 4> state{64, 0, 0};
    for (std::uint32_t i = 0; i < 8; ++i) {
        const auto transition = state.Transition(i * 8, 8, 1U << i, 1U << (i + 8));
        EXPECT_LE(state.IntervalCount(), 4);
        if (i != 0) {
            EXPECT_TRUE(transition.requires_barrier);
        }
    }

    const auto all = state.Transition(0, 64, 0x100, 0x10000);
    EXPECT_EQ(all.prior_access, 0xFF);
    EXPECT_EQ(all.prior_stages, 0xFF00);
}

TEST(BufferAccessRangeState, ConservativeUnionCannotSuppressALaterBarrier) {
    VideoCore::BasicBufferAccessRangeState<std::uint32_t, std::uint32_t, 1> state{64, 0x01, 0x10};
    const auto first = state.Transition(0, 32, 0x02, 0x20);
    ASSERT_TRUE(first.requires_barrier);
    ASSERT_EQ(state.IntervalCount(), 1);

    const auto union_match = state.Transition(0, 64, 0x03, 0x30);
    EXPECT_TRUE(union_match.requires_barrier);
    EXPECT_EQ(union_match.prior_access, 0x03);
    EXPECT_EQ(union_match.prior_stages, 0x30);
}

} // namespace
