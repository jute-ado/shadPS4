// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "imgui/notification_state.h"

TEST(NotificationState, AdvancesNotificationsWithoutReplacingItsOwningLayer) {
    ImGui::NotificationState<int> state;
    auto* const stable_address = &state;

    state.Push(1);
    state.Push(2);

    ASSERT_TRUE(state.ActivateNext());
    ASSERT_NE(state.Current(), nullptr);
    EXPECT_EQ(*state.Current(), 1);

    EXPECT_TRUE(state.CompleteCurrent());
    EXPECT_EQ(&state, stable_address);
    ASSERT_NE(state.Current(), nullptr);
    EXPECT_EQ(*state.Current(), 2);

    EXPECT_FALSE(state.CompleteCurrent());
    EXPECT_EQ(&state, stable_address);
    EXPECT_EQ(state.Current(), nullptr);

    state.Push(3);
    ASSERT_TRUE(state.ActivateNext());
    ASSERT_NE(state.Current(), nullptr);
    EXPECT_EQ(*state.Current(), 3);
}
