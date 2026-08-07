// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <limits>

#include "common/types.h"

namespace AmdGpu {

struct PixelPipeStatControl {
    u32 counter_id{};
    u32 stride_bytes{4};
    u64 instance_mask{};
};

constexpr PixelPipeStatControl DecodePixelPipeStatControl(u32 control_lo, u32 control_hi) {
    return {
        .counter_id = (control_lo >> 3) & 0x3FU,
        .stride_bytes = 4U << ((control_lo >> 9) & 0x3U),
        .instance_mask = static_cast<u64>(control_lo >> 11) | (static_cast<u64>(control_hi) << 21),
    };
}

struct PixelPipeStatLayout {
    std::array<u32, std::numeric_limits<u64>::digits> offsets{};
    u32 count{};
};

constexpr PixelPipeStatLayout BuildPixelPipeStatLayout(const PixelPipeStatControl& control,
                                                       u32 max_instances) {
    PixelPipeStatLayout layout{};
    const u32 bounded_instances =
        max_instances < layout.offsets.size() ? max_instances : layout.offsets.size();
    for (u32 instance = 0; instance < bounded_instances; ++instance) {
        if ((control.instance_mask & (1ULL << instance)) != 0) {
            layout.offsets[layout.count++] = instance * control.stride_bytes;
        }
    }
    return layout;
}

constexpr bool IsFixedPixelPipeStatLayout(const PixelPipeStatControl& control,
                                          u32 fixed_instances) {
    if (fixed_instances == 0 || fixed_instances > std::numeric_limits<u64>::digits ||
        control.stride_bytes != 16) {
        return false;
    }
    const u64 expected_mask = fixed_instances == std::numeric_limits<u64>::digits
                                  ? std::numeric_limits<u64>::max()
                                  : (1ULL << fixed_instances) - 1;
    return control.instance_mask == expected_mask;
}

} // namespace AmdGpu
