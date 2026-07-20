// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "input/controller_axis.h"

TEST(ControllerAxis, ParsesEverySupportedProtocolName) {
    const struct {
        const char* name;
        Input::Axis expected;
    } cases[] = {
        {"left_x", Input::Axis::LeftX},   {"left_y", Input::Axis::LeftY},
        {"right_x", Input::Axis::RightX}, {"right_y", Input::Axis::RightY},
        {"l2", Input::Axis::TriggerLeft}, {"r2", Input::Axis::TriggerRight},
    };

    for (const auto& test : cases) {
        EXPECT_EQ(Input::ParseControllerAxis(test.name), test.expected) << test.name;
    }
}

TEST(ControllerAxis, RejectsUnknownOrIncorrectlyCasedNames) {
    EXPECT_FALSE(Input::ParseControllerAxis("left_z").has_value());
    EXPECT_FALSE(Input::ParseControllerAxis("Left_X").has_value());
    EXPECT_FALSE(Input::ParseControllerAxis("").has_value());
}
