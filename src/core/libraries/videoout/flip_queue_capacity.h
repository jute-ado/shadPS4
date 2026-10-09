// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::VideoOut {

constexpr s32 MaxPendingFlips = 16;

[[nodiscard]] constexpr bool CanQueueFlip(s32 pending_flips, s32 buffer_index) {
    return buffer_index == -1 || pending_flips < MaxPendingFlips;
}

[[nodiscard]] constexpr bool ReserveFlip(s32& pending_flips, s32& eop_flips, s32 buffer_index,
                                         bool is_eop) {
    if (!CanQueueFlip(pending_flips, buffer_index)) {
        return false;
    }
    ++pending_flips;
    if (is_eop) {
        ++eop_flips;
    }
    return true;
}

constexpr void CompleteFlip(s32& pending_flips, s32& eop_flips, bool is_eop) {
    --pending_flips;
    if (is_eop) {
        --eop_flips;
    }
}

} // namespace Libraries::VideoOut
