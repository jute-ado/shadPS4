// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Vulkan {

constexpr u64 NextSubmissionRetirementTick(u64 current_tick) {
    return current_tick + 1;
}

constexpr bool ResourceUseHasRetired(u64 last_use_tick, u64 known_gpu_tick) {
    return known_gpu_tick >= last_use_tick;
}

} // namespace Vulkan
