// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>

#include "common/types.h"

namespace Input {

struct Colour {
    u8 r;
    u8 g;
    u8 b;

    bool operator==(const Colour&) const = default;
};

constexpr std::optional<Colour> LightBarColourForPhysicalReconnect(
    const std::optional<Colour> override_colour) {
    return override_colour;
}

constexpr Colour ResolveLightBarColour(const std::optional<Colour> override_colour,
                                       const Colour requested_colour) {
    return override_colour.value_or(requested_colour);
}

} // namespace Input
