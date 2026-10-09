// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <string_view>

#include "input/controller.h"

namespace Core::Ipc {

constexpr std::optional<Libraries::Pad::OrbisPadButtonDataOffset> ParseTestLabButton(
    const std::string_view name) {
    using Button = Libraries::Pad::OrbisPadButtonDataOffset;
    if (name == "cross") return Button::Cross;
    if (name == "circle") return Button::Circle;
    if (name == "square") return Button::Square;
    if (name == "triangle") return Button::Triangle;
    if (name == "options") return Button::Options;
    if (name == "dpad_up") return Button::Up;
    if (name == "dpad_right") return Button::Right;
    if (name == "dpad_down") return Button::Down;
    if (name == "dpad_left") return Button::Left;
    if (name == "l1") return Button::L1;
    if (name == "l2") return Button::L2;
    if (name == "r1") return Button::R1;
    if (name == "r2") return Button::R2;
    if (name == "l3") return Button::L3;
    if (name == "r3") return Button::R3;
    if (name == "touchpad") return Button::TouchPad;
    return std::nullopt;
}

constexpr std::optional<Input::Axis> ParseTestLabAxis(const std::string_view name) {
    if (name == "left_x") return Input::Axis::LeftX;
    if (name == "left_y") return Input::Axis::LeftY;
    if (name == "right_x") return Input::Axis::RightX;
    if (name == "right_y") return Input::Axis::RightY;
    if (name == "l2") return Input::Axis::TriggerLeft;
    if (name == "r2") return Input::Axis::TriggerRight;
    return std::nullopt;
}

} // namespace Core::Ipc
