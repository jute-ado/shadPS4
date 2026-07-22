// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "input/virtual_gamepad.h"

using Libraries::Pad::OrbisPadButtonDataOffset;

TEST(VirtualGamepad, MapsInjectedButtonsToOverlayNavigation) {
    const struct {
        OrbisPadButtonDataOffset button;
        SDL_GamepadButton expected;
    } cases[] = {
        {OrbisPadButtonDataOffset::Cross, SDL_GAMEPAD_BUTTON_SOUTH},
        {OrbisPadButtonDataOffset::Circle, SDL_GAMEPAD_BUTTON_EAST},
        {OrbisPadButtonDataOffset::Square, SDL_GAMEPAD_BUTTON_WEST},
        {OrbisPadButtonDataOffset::Triangle, SDL_GAMEPAD_BUTTON_NORTH},
        {OrbisPadButtonDataOffset::Options, SDL_GAMEPAD_BUTTON_START},
        {OrbisPadButtonDataOffset::TouchPad, SDL_GAMEPAD_BUTTON_BACK},
        {OrbisPadButtonDataOffset::Up, SDL_GAMEPAD_BUTTON_DPAD_UP},
        {OrbisPadButtonDataOffset::Right, SDL_GAMEPAD_BUTTON_DPAD_RIGHT},
        {OrbisPadButtonDataOffset::Down, SDL_GAMEPAD_BUTTON_DPAD_DOWN},
        {OrbisPadButtonDataOffset::Left, SDL_GAMEPAD_BUTTON_DPAD_LEFT},
        {OrbisPadButtonDataOffset::L1, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER},
        {OrbisPadButtonDataOffset::R1, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER},
        {OrbisPadButtonDataOffset::L3, SDL_GAMEPAD_BUTTON_LEFT_STICK},
        {OrbisPadButtonDataOffset::R3, SDL_GAMEPAD_BUTTON_RIGHT_STICK},
    };

    for (const auto& test : cases) {
        EXPECT_EQ(Input::VirtualButtonToSDLGamepadButton(test.button), test.expected);
    }
}

TEST(VirtualGamepad, DoesNotRouteUnknownButtonBitsToOverlay) {
    EXPECT_FALSE(
        Input::VirtualButtonToSDLGamepadButton(static_cast<OrbisPadButtonDataOffset>(1U << 31))
            .has_value());
}

TEST(VirtualGamepad, PreservesTapUntilOverlaySamplesIt) {
    Input::VirtualGamepadButtonState state;

    state.SetPressed(true);
    state.SetPressed(false);

    EXPECT_TRUE(state.Sample(false));
    EXPECT_FALSE(state.Sample(false));
}

TEST(VirtualGamepad, ReleasesHeldButtonAfterOverlayObservedIt) {
    Input::VirtualGamepadButtonState state;

    state.SetPressed(true);
    EXPECT_TRUE(state.Sample(false));
    EXPECT_TRUE(state.Sample(false));

    state.SetPressed(false);
    EXPECT_FALSE(state.Sample(false));
}

TEST(VirtualGamepad, PhysicalButtonRemainsPressedAfterVirtualRelease) {
    Input::VirtualGamepadButtonState state;

    state.SetPressed(true);
    EXPECT_TRUE(state.Sample(false));
    state.SetPressed(false);

    EXPECT_TRUE(state.Sample(true));
}
