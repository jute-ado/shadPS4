// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>

#include "common/types.h"

namespace Vulkan {

struct ComputeDispatchChunkShape {
    u32 x{};
    u32 y{};
    u32 z{};

    constexpr bool operator==(const ComputeDispatchChunkShape&) const noexcept = default;
};

struct ComputeDispatchChunk3D {
    u32 base_x{};
    u32 base_y{};
    u32 base_z{};
    u32 count_x{};
    u32 count_y{};
    u32 count_z{};

    constexpr bool operator==(const ComputeDispatchChunk3D&) const noexcept = default;
};

[[nodiscard]] constexpr ComputeDispatchChunkShape WatchdogSafeComputeChunkShape(
    bool has_dynamic_scalar_mask_loop, u32 dim_x, u32 dim_y, u32 dim_z,
    u32 workgroup_budget = 1024) noexcept {
    if (!has_dynamic_scalar_mask_loop || dim_x == 0 || dim_y == 0 || dim_z == 0 ||
        workgroup_budget == 0) {
        return {dim_x, dim_y, dim_z};
    }

    u32 remaining = workgroup_budget;
    // Keep genuinely multidimensional tiles spatially bounded. A one-dimensional dispatch has no
    // second spatial axis to amplify a data-dependent mask walk, so let it use the full bounded
    // workgroup budget instead of manufacturing eight times as many submissions.
    const bool compact_budget = workgroup_budget <= 512;
    const u32 spatial_x_limit = compact_budget ? 16U : 64U;
    const u32 spatial_y_limit = compact_budget ? 8U : 32U;
    const u32 max_chunk_x =
        dim_y == 1 && dim_z == 1 ? remaining : std::min(spatial_x_limit, remaining);
    const u32 chunk_x = std::min(dim_x, max_chunk_x);
    remaining = std::max(remaining / chunk_x, 1U);
    const u32 chunk_y = std::min(dim_y, std::min(spatial_y_limit, remaining));
    remaining = std::max(remaining / chunk_y, 1U);
    const u32 chunk_z = std::min(dim_z, remaining);
    return {chunk_x, chunk_y, chunk_z};
}

[[nodiscard]] constexpr u32 ComputeDispatchChunkCount(u32 dimension, u32 chunk) noexcept {
    return dimension == 0 || chunk == 0 ? 0 : 1 + (dimension - 1) / chunk;
}

[[nodiscard]] constexpr u32 ComputeDispatchChunk3DCount(
    u32 dim_x, u32 dim_y, u32 dim_z, ComputeDispatchChunkShape shape) noexcept {
    return ComputeDispatchChunkCount(dim_x, shape.x) *
           ComputeDispatchChunkCount(dim_y, shape.y) *
           ComputeDispatchChunkCount(dim_z, shape.z);
}

[[nodiscard]] constexpr ComputeDispatchChunk3D ComputeDispatchChunk3DAt(
    u32 dim_x, u32 dim_y, u32 dim_z, ComputeDispatchChunkShape shape,
    u32 chunk_index) noexcept {
    const u32 chunks_x = ComputeDispatchChunkCount(dim_x, shape.x);
    const u32 chunks_y = ComputeDispatchChunkCount(dim_y, shape.y);
    const u32 chunks_z = ComputeDispatchChunkCount(dim_z, shape.z);
    const u32 chunk_count = chunks_x * chunks_y * chunks_z;
    if (chunk_index >= chunk_count || chunk_count == 0) {
        return {};
    }

    const u32 chunk_x = chunk_index % chunks_x;
    const u32 chunk_y = (chunk_index / chunks_x) % chunks_y;
    const u32 chunk_z = chunk_index / (chunks_x * chunks_y);
    const u32 base_x = chunk_x * shape.x;
    const u32 base_y = chunk_y * shape.y;
    const u32 base_z = chunk_z * shape.z;
    return {
        .base_x = base_x,
        .base_y = base_y,
        .base_z = base_z,
        .count_x = std::min(shape.x, dim_x - base_x),
        .count_y = std::min(shape.y, dim_y - base_y),
        .count_z = std::min(shape.z, dim_z - base_z),
    };
}

} // namespace Vulkan
