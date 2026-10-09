// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "input/controller_lightbar.h"

TEST(ControllerLightBar, PhysicalReconnectReappliesOverrideColour) {
    constexpr Input::Colour override_colour{17, 34, 51};

    const auto reconnect_colour = Input::LightBarColourForPhysicalReconnect(override_colour);

    ASSERT_TRUE(reconnect_colour.has_value());
    EXPECT_EQ(*reconnect_colour, override_colour);
}

TEST(ControllerLightBar, PhysicalReconnectWithoutOverrideDoesNotReplaceCurrentColour) {
    EXPECT_FALSE(Input::LightBarColourForPhysicalReconnect(std::nullopt).has_value());
}

TEST(ControllerLightBar, OverrideWinsWhenGameRequestsAnotherColour) {
    constexpr Input::Colour override_colour{17, 34, 51};
    constexpr Input::Colour requested_colour{68, 85, 102};

    EXPECT_EQ(Input::ResolveLightBarColour(override_colour, requested_colour), override_colour);
    EXPECT_EQ(Input::ResolveLightBarColour(std::nullopt, requested_colour), requested_colour);
}
