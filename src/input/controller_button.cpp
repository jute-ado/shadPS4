// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "controller_button.h"

#include <array>
#include <utility>

namespace Input {

std::optional<Libraries::Pad::OrbisPadButtonDataOffset> ParseControllerButton(
    std::string_view name) {
    using namespace std::string_view_literals;
    using Button = Libraries::Pad::OrbisPadButtonDataOffset;
    static constexpr std::array mappings{
        std::pair{"cross"sv, Button::Cross},
        std::pair{"circle"sv, Button::Circle},
        std::pair{"square"sv, Button::Square},
        std::pair{"triangle"sv, Button::Triangle},
        std::pair{"options"sv, Button::Options},
        std::pair{"dpad_up"sv, Button::Up},
        std::pair{"dpad_right"sv, Button::Right},
        std::pair{"dpad_down"sv, Button::Down},
        std::pair{"dpad_left"sv, Button::Left},
        std::pair{"l1"sv, Button::L1},
        std::pair{"l2"sv, Button::L2},
        std::pair{"r1"sv, Button::R1},
        std::pair{"r2"sv, Button::R2},
        std::pair{"l3"sv, Button::L3},
        std::pair{"r3"sv, Button::R3},
        std::pair{"touchpad"sv, Button::TouchPad},
    };
    for (const auto& [mapping_name, button] : mappings) {
        if (name == mapping_name) {
            return button;
        }
    }
    return std::nullopt;
}

} // namespace Input
