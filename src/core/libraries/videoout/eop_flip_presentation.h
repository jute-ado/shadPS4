// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::VideoOut {

[[nodiscard]] constexpr bool IsFlipPresentationEligible(bool is_eop, u64 submitted_vblank,
                                                        u64 current_vblank) noexcept {
    return !is_eop || current_vblank > submitted_vblank;
}

} // namespace Libraries::VideoOut
