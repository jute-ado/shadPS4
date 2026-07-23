// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "input/controller_button.h"

using Libraries::Pad::OrbisPadButtonDataOffset;

TEST(ControllerButton, ParsesEverySupportedProtocolName) {
    const struct {
        const char* name;
        OrbisPadButtonDataOffset expected;
    } cases[] = {
        {"cross", OrbisPadButtonDataOffset::Cross},
        {"circle", OrbisPadButtonDataOffset::Circle},
        {"square", OrbisPadButtonDataOffset::Square},
        {"triangle", OrbisPadButtonDataOffset::Triangle},
        {"options", OrbisPadButtonDataOffset::Options},
        {"dpad_up", OrbisPadButtonDataOffset::Up},
        {"dpad_right", OrbisPadButtonDataOffset::Right},
        {"dpad_down", OrbisPadButtonDataOffset::Down},
        {"dpad_left", OrbisPadButtonDataOffset::Left},
        {"l1", OrbisPadButtonDataOffset::L1},
        {"l2", OrbisPadButtonDataOffset::L2},
        {"r1", OrbisPadButtonDataOffset::R1},
        {"r2", OrbisPadButtonDataOffset::R2},
        {"l3", OrbisPadButtonDataOffset::L3},
        {"r3", OrbisPadButtonDataOffset::R3},
        {"touchpad", OrbisPadButtonDataOffset::TouchPad},
    };

    for (const auto& test : cases) {
        EXPECT_EQ(Input::ParseControllerButton(test.name), test.expected) << test.name;
    }
}

TEST(ControllerButton, RejectsUnknownOrIncorrectlyCasedNames) {
    EXPECT_FALSE(Input::ParseControllerButton("start").has_value());
    EXPECT_FALSE(Input::ParseControllerButton("Cross").has_value());
    EXPECT_FALSE(Input::ParseControllerButton("").has_value());
}
