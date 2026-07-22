// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <optional>

#include <SDL3/SDL_gamepad.h>
#include "core/libraries/pad/pad.h"

namespace Input {

constexpr std::optional<SDL_GamepadButton> VirtualButtonToSDLGamepadButton(
    Libraries::Pad::OrbisPadButtonDataOffset button) {
    using Button = Libraries::Pad::OrbisPadButtonDataOffset;
    switch (button) {
    case Button::Cross:
        return SDL_GAMEPAD_BUTTON_SOUTH;
    case Button::Circle:
        return SDL_GAMEPAD_BUTTON_EAST;
    case Button::Square:
        return SDL_GAMEPAD_BUTTON_WEST;
    case Button::Triangle:
        return SDL_GAMEPAD_BUTTON_NORTH;
    case Button::Options:
        return SDL_GAMEPAD_BUTTON_START;
    case Button::TouchPad:
        return SDL_GAMEPAD_BUTTON_BACK;
    case Button::Up:
        return SDL_GAMEPAD_BUTTON_DPAD_UP;
    case Button::Right:
        return SDL_GAMEPAD_BUTTON_DPAD_RIGHT;
    case Button::Down:
        return SDL_GAMEPAD_BUTTON_DPAD_DOWN;
    case Button::Left:
        return SDL_GAMEPAD_BUTTON_DPAD_LEFT;
    case Button::L1:
        return SDL_GAMEPAD_BUTTON_LEFT_SHOULDER;
    case Button::R1:
        return SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER;
    case Button::L3:
        return SDL_GAMEPAD_BUTTON_LEFT_STICK;
    case Button::R3:
        return SDL_GAMEPAD_BUTTON_RIGHT_STICK;
    default:
        return std::nullopt;
    }
}

class VirtualGamepadButtonState {
public:
    void SetPressed(bool pressed) {
        if (pressed) {
            state.store(State::PressedUnobserved, std::memory_order_relaxed);
            return;
        }

        auto current = state.load(std::memory_order_relaxed);
        while (current != State::Released && current != State::ReleasePending) {
            const auto desired =
                current == State::PressedUnobserved ? State::ReleasePending : State::Released;
            if (state.compare_exchange_weak(current, desired, std::memory_order_relaxed)) {
                return;
            }
        }
    }

    bool Sample(bool physical_pressed) {
        auto current = state.load(std::memory_order_relaxed);
        while (true) {
            switch (current) {
            case State::Released:
                return physical_pressed;
            case State::PressedObserved:
                return true;
            case State::PressedUnobserved:
                if (state.compare_exchange_weak(current, State::PressedObserved,
                                                std::memory_order_relaxed)) {
                    return true;
                }
                break;
            case State::ReleasePending:
                if (state.compare_exchange_weak(current, State::Released,
                                                std::memory_order_relaxed)) {
                    return true;
                }
                break;
            }
        }
    }

private:
    enum class State {
        Released,
        PressedUnobserved,
        PressedObserved,
        ReleasePending,
    };

    std::atomic<State> state{State::Released};
};

} // namespace Input
