// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/pad/pad.h"

namespace Input {

struct HostNavigationButtons {
    bool start{};
    bool face_right{};
    bool face_up{};
    bool face_down{};
    bool dpad_left{};
    bool dpad_right{};
    bool dpad_up{};
    bool dpad_down{};
    bool l1{};
    bool r1{};
    bool l3{};
    bool r3{};

    bool operator==(const HostNavigationButtons&) const = default;
};

[[nodiscard]] constexpr HostNavigationButtons GetHostNavigationButtons(
    Libraries::Pad::OrbisPadButtonDataOffset buttons) {
    using Libraries::Pad::OrbisPadButtonDataOffset;
    const auto pressed = [buttons](OrbisPadButtonDataOffset button) {
        return (buttons & button) != OrbisPadButtonDataOffset::None;
    };
    return {
        .start = pressed(OrbisPadButtonDataOffset::Options),
        .face_right = pressed(OrbisPadButtonDataOffset::Circle),
        .face_up = pressed(OrbisPadButtonDataOffset::Triangle),
        .face_down = pressed(OrbisPadButtonDataOffset::Cross),
        .dpad_left = pressed(OrbisPadButtonDataOffset::Left),
        .dpad_right = pressed(OrbisPadButtonDataOffset::Right),
        .dpad_up = pressed(OrbisPadButtonDataOffset::Up),
        .dpad_down = pressed(OrbisPadButtonDataOffset::Down),
        .l1 = pressed(OrbisPadButtonDataOffset::L1),
        .r1 = pressed(OrbisPadButtonDataOffset::R1),
        .l3 = pressed(OrbisPadButtonDataOffset::L3),
        .r3 = pressed(OrbisPadButtonDataOffset::R3),
    };
}

} // namespace Input
