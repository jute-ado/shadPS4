// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdint>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/buffer_access_interval_provenance.h"

namespace {

using Tracker = VideoCore::BasicBufferAccessIntervalProvenance<std::uint32_t>;
using Role = VideoCore::BufferAccessRole;
using RangeState = VideoCore::BasicBufferAccessRangeState<std::uint32_t>;

TEST(BufferAccessIntervalProvenance, DisjointRolesKeepThePriorProducerAndOverlapAggregates) {
    Tracker tracker;

    tracker.BeginCommand(11, 50);
    tracker.Observe(7, 0, 128, Role::TransferWrite, 0x01, 0x10, true);
    ASSERT_EQ(tracker.CommitCommand().size(), 1);

    tracker.BeginCommand(12, 51);
    tracker.Observe(7, 0, 64, Role::ShaderReadWrite, 0x06, 0x20, true);
    tracker.Observe(7, 64, 64, Role::VertexRead, 0x08, 0x40, false);
    tracker.Observe(7, 64, 64, Role::IndexRead, 0x10, 0x80, false);

    const auto transitions = tracker.CommitCommand();
    ASSERT_EQ(transitions.size(), 2);

    const auto& shader = transitions[0];
    EXPECT_EQ(shader.offset, 0);
    EXPECT_EQ(shader.size, 64);
    ASSERT_TRUE(shader.prior.has_value());
    EXPECT_EQ(shader.prior->command_id, 11);
    EXPECT_EQ(shader.prior->tick, 50);
    EXPECT_EQ(shader.prior->roles, Role::TransferWrite);
    EXPECT_EQ(shader.observations.size(), 1);
    EXPECT_EQ(shader.current_roles, Role::ShaderReadWrite);

    const auto& geometry = transitions[1];
    EXPECT_EQ(geometry.offset, 64);
    EXPECT_EQ(geometry.size, 64);
    ASSERT_TRUE(geometry.prior.has_value());
    EXPECT_EQ(geometry.prior->command_id, 11);
    EXPECT_EQ(geometry.prior->tick, 50);
    EXPECT_EQ(geometry.prior->roles, Role::TransferWrite);
    EXPECT_EQ(geometry.observations.size(), 2);
    EXPECT_EQ(geometry.current_roles, Role::VertexRead | Role::IndexRead);
    EXPECT_EQ(geometry.current_access, 0x18);
    EXPECT_EQ(geometry.current_stages, 0xC0);
}

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

TEST(BufferAccessRangeState, OverlapUnionsOnlyTheCoveredPriorIntervals) {
    RangeState state{192, 0x01, 0x10};
    state.Transition(0, 64, 0x02, 0x20);
    state.Transition(128, 64, 0x04, 0x40);

    const auto overlap = state.Transition(32, 128, 0x08, 0x80);

    EXPECT_TRUE(overlap.requires_barrier);
    EXPECT_EQ(overlap.prior_access, 0x01 | 0x02 | 0x04);
    EXPECT_EQ(overlap.prior_stages, 0x10 | 0x20 | 0x40);
}

} // namespace
