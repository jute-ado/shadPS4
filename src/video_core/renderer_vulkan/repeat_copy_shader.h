// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include "common/types.h"

namespace Vulkan {

constexpr bool CanSkipRepeatCopyDispatch(u32 control_count, u32 user_count,
                                         u32 dispatch_capacity, bool control_gpu_owned) {
    if (control_gpu_owned) {
        return false;
    }
    return std::min({control_count, user_count, dispatch_capacity}) == 0;
}

} // namespace Vulkan
