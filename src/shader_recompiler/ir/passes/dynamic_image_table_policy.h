// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bit>
#include <optional>

#include "common/types.h"

namespace Shader::Optimization {

struct DynamicImageTablePattern {
    u32 base_pointer_location{};
    u32 byte_offset{};
    u32 byte_stride{};
    bool is_r128{};
    bool supports_nonuniform_indexing{};
};

struct DynamicImageTablePlan {
    u32 base_pointer_location{};
    u32 offset_dw{};
    u32 stride_dw{};
    u32 descriptor_count{};
    u32 index_shift{};
};

[[nodiscard]] constexpr std::optional<DynamicImageTablePlan> PlanDynamicImageTable(
    const DynamicImageTablePattern pattern) {
    constexpr u32 MaxDynamicImageDescriptors = 64;
    if (!pattern.supports_nonuniform_indexing || pattern.byte_offset % sizeof(u32) != 0 ||
        pattern.byte_stride % sizeof(u32) != 0) {
        return std::nullopt;
    }
    const u32 stride_dw = pattern.byte_stride / sizeof(u32);
    const u32 descriptor_dw = pattern.is_r128 ? 4u : 8u;
    if (stride_dw < descriptor_dw || !std::has_single_bit(stride_dw)) {
        return std::nullopt;
    }
    return DynamicImageTablePlan{
        .base_pointer_location = pattern.base_pointer_location,
        .offset_dw = static_cast<u32>(pattern.byte_offset / sizeof(u32)),
        .stride_dw = stride_dw,
        .descriptor_count = MaxDynamicImageDescriptors,
        .index_shift = static_cast<u32>(std::countr_zero(stride_dw)),
    };
}

} // namespace Shader::Optimization
