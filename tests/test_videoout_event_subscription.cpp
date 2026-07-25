// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <vector>

#include <gtest/gtest.h>

#include "core/libraries/videoout/videoout_event_subscription.h"

namespace Libraries::VideoOut {
namespace {

TEST(VideoOutEventSubscription, DeletionKeyMatchesTheInternalRegistrationKey) {
    EXPECT_EQ(GetVideoOutEventIdent(OrbisVideoOutInternalEventId::Flip), 0x6u);
    EXPECT_EQ(GetVideoOutEventIdent(OrbisVideoOutInternalEventId::Vblank), 0x7u);

    constexpr s32 main_port_handle = 1;
    EXPECT_NE(GetVideoOutEventIdent(OrbisVideoOutInternalEventId::Flip), main_port_handle);
    EXPECT_NE(GetVideoOutEventIdent(OrbisVideoOutInternalEventId::Vblank), main_port_handle);
}

TEST(VideoOutEventSubscription, RepeatedRegistrationDoesNotDuplicateNotifications) {
    std::vector<s64> subscriptions;

    AddVideoOutEventSubscription(subscriptions, 10);
    AddVideoOutEventSubscription(subscriptions, 10);
    AddVideoOutEventSubscription(subscriptions, 11);

    EXPECT_EQ(subscriptions, (std::vector<s64>{10, 11}));
}

TEST(VideoOutEventSubscription, DeletionRemovesTheRequestedQueueOnly) {
    std::vector<s64> subscriptions{10, 11, 12};

    EXPECT_TRUE(RemoveVideoOutEventSubscription(subscriptions, 11));
    EXPECT_EQ(subscriptions, (std::vector<s64>{10, 12}));
    EXPECT_FALSE(RemoveVideoOutEventSubscription(subscriptions, 99));
    EXPECT_EQ(subscriptions, (std::vector<s64>{10, 12}));
}

} // namespace
} // namespace Libraries::VideoOut
