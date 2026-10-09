// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <compare>
#include <initializer_list>
#include "common/types.h"

namespace Shader {

// Command-processor destinations are register locations, not per-draw values. Keep the disabled
// sentinel separate from SGPR0, which is a valid destination when selected by a packet.
struct DrawUserDataLocations {
    static constexpr u8 Disabled = 0xffU;
    static constexpr u32 NumRegisters = 16U;
    u8 base_vertex = Disabled;
    u8 base_instance = Disabled;
    u8 draw_index = Disabled;

    [[nodiscard]] constexpr bool IsValid() const noexcept {
        const auto valid = [](u8 value) { return value == Disabled || value < NumRegisters; };
        return valid(base_vertex) && valid(base_instance) && valid(draw_index) &&
               (base_vertex == Disabled || base_vertex != base_instance) &&
               (base_vertex == Disabled || base_vertex != draw_index) &&
               (base_instance == Disabled || base_instance != draw_index);
    }

    [[nodiscard]] constexpr u32 RegisterMask() const noexcept {
        u32 mask{};
        for (const u8 location : {base_vertex, base_instance, draw_index}) {
            if (location < NumRegisters) {
                mask |= 1U << location;
            }
        }
        return mask;
    }

    [[nodiscard]] constexpr bool OverlapsSpan(u32 first, u32 count) const noexcept {
        for (const u8 location : {base_vertex, base_instance, draw_index}) {
            if (location < NumRegisters && location >= first && u32(location) - first < count) {
                return true;
            }
        }
        return false;
    }

    auto operator<=>(const DrawUserDataLocations&) const noexcept = default;
};

} // namespace Shader
