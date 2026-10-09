// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/kernel/threads/guest_thread_priority.h"

using Common::ThreadPriority;
using Libraries::Kernel::MapGuestThreadPriority;
using Libraries::Kernel::SchedPolicy;

TEST(GuestThreadPriority, TimeSharingThreadsRemainAtNormalHostPriority) {
    EXPECT_EQ(MapGuestThreadPriority(SchedPolicy::Other, 260), ThreadPriority::Normal);
    EXPECT_EQ(MapGuestThreadPriority(SchedPolicy::Other, 900), ThreadPriority::Normal);
}

TEST(GuestThreadPriority, RealtimeDefaultMapsToNormalHostPriority) {
    EXPECT_EQ(MapGuestThreadPriority(SchedPolicy::Fifo, 700), ThreadPriority::Normal);
    EXPECT_EQ(MapGuestThreadPriority(SchedPolicy::RoundRobin, 700), ThreadPriority::Normal);
}

TEST(GuestThreadPriority, LowerRealtimeValuesMapAboveTheDefault) {
    EXPECT_EQ(MapGuestThreadPriority(SchedPolicy::Fifo, 699), ThreadPriority::High);
    EXPECT_EQ(MapGuestThreadPriority(SchedPolicy::RoundRobin, 256), ThreadPriority::High);
}

TEST(GuestThreadPriority, HigherRealtimeValuesMapBelowTheDefault) {
    EXPECT_EQ(MapGuestThreadPriority(SchedPolicy::Fifo, 701), ThreadPriority::Low);
    EXPECT_EQ(MapGuestThreadPriority(SchedPolicy::RoundRobin, 767), ThreadPriority::Low);
}
