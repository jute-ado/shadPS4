// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "input/controller_button.h"
#include "input/host_navigation_buttons.h"

using Libraries::Pad::OrbisPadButtonDataOffset;

TEST(ControllerButton, ParsesEveryProtocolName) {
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

TEST(ControllerButton, RejectsUnknownNames) {
    EXPECT_FALSE(Input::ParseControllerButton("start").has_value());
    EXPECT_FALSE(Input::ParseControllerButton("Cross").has_value());
    EXPECT_FALSE(Input::ParseControllerButton("").has_value());
}

TEST(HostNavigationButtons, MapsVirtualGuestPadButtonsToTheHostDialog) {
    const auto buttons = OrbisPadButtonDataOffset::Options | OrbisPadButtonDataOffset::Cross |
                         OrbisPadButtonDataOffset::Circle | OrbisPadButtonDataOffset::Triangle |
                         OrbisPadButtonDataOffset::Up | OrbisPadButtonDataOffset::Right |
                         OrbisPadButtonDataOffset::Down | OrbisPadButtonDataOffset::Left |
                         OrbisPadButtonDataOffset::L1 | OrbisPadButtonDataOffset::R1 |
                         OrbisPadButtonDataOffset::L3 | OrbisPadButtonDataOffset::R3;

    EXPECT_EQ(Input::GetHostNavigationButtons(buttons),
              (Input::HostNavigationButtons{
                  .start = true,
                  .face_right = true,
                  .face_up = true,
                  .face_down = true,
                  .dpad_left = true,
                  .dpad_right = true,
                  .dpad_up = true,
                  .dpad_down = true,
                  .l1 = true,
                  .r1 = true,
                  .l3 = true,
                  .r3 = true,
              }));
}

TEST(HostNavigationButtons, ReleasesEveryHostDialogButtonAtNeutral) {
    EXPECT_EQ(Input::GetHostNavigationButtons(OrbisPadButtonDataOffset::None),
              Input::HostNavigationButtons{});
}
