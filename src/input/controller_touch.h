// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Input {

constexpr u16 ControllerTouchMaxX = 1919;
constexpr u16 ControllerTouchMaxY = 941;

struct ControllerTouchEvent {
    u8 finger;
    bool down;
    u16 x;
    u16 y;
};

[[nodiscard]] constexpr bool IsValidControllerTouch(u64 finger, u64 x, u64 y) {
    return finger < 2 && x <= ControllerTouchMaxX && y <= ControllerTouchMaxY;
}

[[nodiscard]] constexpr u64 PackControllerTouch(u16 x, u16 y, bool down) {
    return static_cast<u64>(x) | (static_cast<u64>(y) << 11) | (static_cast<u64>(down) << 21);
}

[[nodiscard]] constexpr ControllerTouchEvent UnpackControllerTouch(u8 finger, u64 packed) {
    return {
        .finger = finger,
        .down = ((packed >> 21) & 1) != 0,
        .x = static_cast<u16>(packed & 0x7ff),
        .y = static_cast<u16>((packed >> 11) & 0x3ff),
    };
}

} // namespace Input
