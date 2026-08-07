// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bit>

#include "common/types.h"

namespace AmdGpu {

struct PixelPipeStatControl {
    static constexpr u32 CounterIdShift = 3;
    static constexpr u32 CounterIdMask = 0x3F;
    static constexpr u32 StrideShift = 9;
    static constexpr u32 StrideMask = 0x3;
    static constexpr u32 InstanceMaskShift = 11;
    static constexpr u32 InstanceMaskLowBits = 21;

    [[nodiscard]] static constexpr PixelPipeStatControl Decode(u32 low, u32 high) noexcept {
        const u32 stride_encoding = (low >> StrideShift) & StrideMask;
        return {
            .counter_id = (low >> CounterIdShift) & CounterIdMask,
            .stride_bytes = 4U << stride_encoding,
            .instance_enable_mask =
                (static_cast<u64>(low >> InstanceMaskShift) |
                 (static_cast<u64>(high) << InstanceMaskLowBits)),
        };
    }

    [[nodiscard]] constexpr u32 EnabledInstanceCount() const noexcept {
        return std::popcount(instance_enable_mask);
    }

    template <typename Callback>
    constexpr void ForEachEnabledInstance(u32 instance_limit, Callback&& callback) const {
        for (u32 instance = 0; instance < instance_limit; ++instance) {
            if ((instance_enable_mask & (1ULL << instance)) != 0) {
                callback(instance, instance * stride_bytes);
            }
        }
    }

    u32 counter_id{};
    u32 stride_bytes{};
    u64 instance_enable_mask{};
};

} // namespace AmdGpu
