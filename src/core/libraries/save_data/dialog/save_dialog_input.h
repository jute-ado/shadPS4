// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>

#include "common/types.h"

namespace Libraries::SaveData::Dialog {

constexpr bool IsNewButtonPress(u32 current_buttons, u32 previous_buttons, u32 button_mask) {
    return (current_buttons & button_mask) != 0 && (previous_buttons & button_mask) == 0;
}

constexpr bool IsNewButtonPressAfterInitialization(bool input_initialized, u32 current_buttons,
                                                   u32 previous_buttons, u32 button_mask) {
    return input_initialized && IsNewButtonPress(current_buttons, previous_buttons, button_mask);
}

constexpr int MoveSelection(int current, int item_count, int delta) {
    if (item_count <= 0) {
        return -1;
    }
    return std::clamp(current + delta, 0, item_count - 1);
}

} // namespace Libraries::SaveData::Dialog
