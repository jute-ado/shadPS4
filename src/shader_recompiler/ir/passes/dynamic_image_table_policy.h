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
    // This is a host-supported descriptor window, not a claim about guest allocation size.
    u32 supported_descriptor_window{};
    bool supports_nonuniform_indexing{};
    bool supports_oob_guard{};
};

struct DynamicImageTablePlan {
    u32 base_pointer_location{};
    u32 offset_dw{};
    u32 stride_dw{};
    u32 descriptor_count{};
    u32 snapshot_span_dw{};
    bool requires_oob_guard{};
};

[[nodiscard]] constexpr u32 DynamicImageDescriptorWindow(
    u32 maximum_window, u64 sampled_descriptors_used, u64 resources_used,
    u32 max_sampled_images_per_stage, u32 max_sampled_images_per_set,
    u32 max_resources_per_stage, u32 max_sampled_images_per_descriptor_heap_set) noexcept {
    const auto remaining = [](u32 limit, u64 used) -> u32 {
        return used >= limit ? 0U : static_cast<u32>(u64{limit} - used);
    };
    return std::min({maximum_window,
                     remaining(max_sampled_images_per_stage, sampled_descriptors_used),
                     remaining(max_sampled_images_per_set, sampled_descriptors_used),
                     remaining(max_resources_per_stage, resources_used),
                     remaining(max_sampled_images_per_descriptor_heap_set,
                               sampled_descriptors_used)});
}

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

    const u64 last_guest_byte_address =
        u64{pattern.byte_offset} + u64{descriptor_count - 1U} * pattern.byte_stride;
    if (last_guest_byte_address > std::numeric_limits<u32>::max()) {
        return std::nullopt;
    }

    const u32 stride_dw = pattern.byte_stride / sizeof(u32);
    const u32 descriptor_dw = pattern.is_r128 ? 4U : 8U;
    if (stride_dw < descriptor_dw || !std::has_single_bit(stride_dw)) {
        return std::nullopt;
    }
    const u32 offset_dw = pattern.byte_offset / sizeof(u32);
    const u64 snapshot_span_dw = u64{descriptor_count - 1U} * stride_dw + descriptor_dw;
    if (snapshot_span_dw > std::numeric_limits<u32>::max() ||
        offset_dw > std::numeric_limits<u32>::max() - snapshot_span_dw) {
        return std::nullopt;
    }

    return DynamicImageTablePlan{
        .base_pointer_location = pattern.base_pointer_location,
        .offset_dw = offset_dw,
        .stride_dw = stride_dw,
        .descriptor_count = descriptor_count,
        .snapshot_span_dw = static_cast<u32>(snapshot_span_dw),
        .requires_oob_guard = requires_oob_guard,
    };
}

} // namespace Shader::Optimization
