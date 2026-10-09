// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Shader {

struct MsaaImageArgumentPlan {
    bool valid{};
    bool drop_zero_mip{};
    bool force_zero_sample{};
    u32 sample_argument{};
};

[[nodiscard]] constexpr MsaaImageArgumentPlan PlanMsaaImageArgument(
    u32 coordinate_arguments, u32 available_arguments, bool has_lod,
    bool lod_is_immediate_zero) {
    if ((coordinate_arguments != 2 && coordinate_arguments != 3) ||
        coordinate_arguments >= available_arguments) {
        return {};
    }
    if (!has_lod) {
        return {.valid = true, .sample_argument = coordinate_arguments};
    }
    if (!lod_is_immediate_zero) {
        return {};
    }
    const u32 following_argument = coordinate_arguments + 1;
    if (following_argument < available_arguments) {
        return {
            .valid = true,
            .drop_zero_mip = true,
            .sample_argument = following_argument,
        };
    }

    // IMAGE_*_MIP has no legal MSAA form in the GCN ISA. For the array case all four
    // translated address operands are already occupied by x, y, slice and the required zero mip.
    // Preserve the bounded zero-mip interpretation as sample zero instead of reading Arg(4).
    return {
        .valid = true,
        .drop_zero_mip = true,
        .force_zero_sample = true,
        .sample_argument = coordinate_arguments,
    };
}

} // namespace Shader
