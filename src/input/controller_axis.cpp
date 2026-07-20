// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "controller_axis.h"

#include <array>
#include <utility>

namespace Input {

std::optional<Axis> ParseControllerAxis(std::string_view name) {
    using namespace std::string_view_literals;
    static constexpr std::array mappings{
        std::pair{"left_x"sv, Axis::LeftX},   std::pair{"left_y"sv, Axis::LeftY},
        std::pair{"right_x"sv, Axis::RightX}, std::pair{"right_y"sv, Axis::RightY},
        std::pair{"l2"sv, Axis::TriggerLeft}, std::pair{"r2"sv, Axis::TriggerRight},
    };

    for (const auto& [protocol_name, axis] : mappings) {
        if (name == protocol_name) {
            return axis;
        }
    }
    return std::nullopt;
}

} // namespace Input
