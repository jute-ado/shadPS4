// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/videoout/vblank_tick.h"

namespace Libraries::VideoOut {
namespace {

TEST(VblankTick, FirstEventPublishesTheFirstCompletedVblank) {
    constexpr auto tick = PrepareVblankTick(0, 1);

    EXPECT_EQ(tick.count, 1u);
    EXPECT_EQ(tick.event_data >> 16, 1u);
}

TEST(VblankTick, EventPayloadMatchesTheNewStatusCount) {
    constexpr u64 prior_count = 42;
    constexpr u64 event_id = 7;
    constexpr auto tick = PrepareVblankTick(prior_count, event_id);

    EXPECT_EQ(tick.count, prior_count + 1);
    EXPECT_EQ(tick.event_data, event_id | (tick.count << 16));
}

} // namespace
} // namespace Libraries::VideoOut
