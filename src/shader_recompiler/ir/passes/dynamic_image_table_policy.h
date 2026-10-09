// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <limits>
#include <optional>

#include "common/types.h"

namespace Shader::Optimization {

enum class DynamicImageAccess : u8 {
    SampledExplicitLod,
    SampledImplicitLod,
    StorageRead,
    StorageWrite,
    Atomic,
};

struct DynamicImageTablePattern {
    u32 base_pointer_location{};
    u32 byte_offset{};
    u32 byte_stride{};
    bool is_r128{};
    DynamicImageAccess access{};
    std::optional<u32> index_upper_bound{};
    // This is the backend resource window we can reserve and guard, not an assertion about the
    // size of the guest table. An unbounded guest index is accepted only when the backend can
    // emit a guard that returns a null sample outside this window.
    u32 supported_descriptor_window{};
    bool supports_nonuniform_indexing{};
    bool supports_oob_guard{};
};

struct DynamicImageTablePlan {
    u32 base_pointer_location{};
    u32 offset_dw{};
    u32 stride_dw{};
    u32 descriptor_count{};
    bool requires_oob_guard{};
};

[[nodiscard]] constexpr std::optional<DynamicImageTablePlan> PlanDynamicImageTable(
    const DynamicImageTablePattern& pattern) {
    if (pattern.access != DynamicImageAccess::SampledExplicitLod ||
        !pattern.supports_nonuniform_indexing || pattern.supported_descriptor_window == 0 ||
        pattern.byte_offset % sizeof(u32) != 0 || pattern.byte_stride % sizeof(u32) != 0) {
        return std::nullopt;
    }

    const bool is_unbounded = !pattern.index_upper_bound.has_value();
    if (is_unbounded && !pattern.supports_oob_guard) {
        return std::nullopt;
    }

    u32 descriptor_count = pattern.supported_descriptor_window;
    bool requires_oob_guard = is_unbounded;
    if (!is_unbounded) {
        if (*pattern.index_upper_bound == std::numeric_limits<u32>::max()) {
            return std::nullopt;
        }
        const u32 proven_descriptor_count = *pattern.index_upper_bound + 1;
        descriptor_count = std::min(proven_descriptor_count, pattern.supported_descriptor_window);
        requires_oob_guard = proven_descriptor_count > pattern.supported_descriptor_window;
        if (requires_oob_guard && !pattern.supports_oob_guard) {
            return std::nullopt;
        }
    }

    const u32 stride_dw = pattern.byte_stride / sizeof(u32);
    const u32 descriptor_dw = pattern.is_r128 ? 4u : 8u;
    if (stride_dw < descriptor_dw || !std::has_single_bit(stride_dw)) {
        return std::nullopt;
    }

    return DynamicImageTablePlan{
        .base_pointer_location = pattern.base_pointer_location,
        .offset_dw = pattern.byte_offset / static_cast<u32>(sizeof(u32)),
        .stride_dw = stride_dw,
        .descriptor_count = descriptor_count,
        .requires_oob_guard = requires_oob_guard,
    };
}

} // namespace Shader::Optimization
