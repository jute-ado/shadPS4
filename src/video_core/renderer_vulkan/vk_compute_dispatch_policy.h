// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>

#include "common/types.h"

namespace Vulkan {

[[nodiscard]] constexpr bool IsComputeDispatchWithinHostLimits(
    const u32 dim_x, const u32 dim_y, const u32 dim_z,
    const std::array<u32, 3>& max_group_count) noexcept {
    return dim_x != 0 && dim_y != 0 && dim_z != 0 && dim_x <= max_group_count[0] &&
           dim_y <= max_group_count[1] && dim_z <= max_group_count[2];
}

[[nodiscard]] constexpr bool ComputeDispatchNeedsBasePipeline(u64 shader_hash) noexcept {
    return shader_hash == 0x87cf9b8a || shader_hash == 0x950e34f8 ||
           shader_hash == 0x50784801 ||
           shader_hash == 0x4defcf7f ||
           shader_hash == 0xbf4600a0 || shader_hash == 0x19de6f57 ||
           shader_hash == 0x57bd9f27;
}

[[nodiscard]] constexpr bool ComputeDispatchRequiresChunkSubmissions(u64 shader_hash) noexcept {
    return shader_hash == 0x87cf9b8a || shader_hash == 0x950e34f8 ||
           shader_hash == 0x50784801 ||
           shader_hash == 0x4defcf7f ||
           shader_hash == 0xbf4600a0 || shader_hash == 0x19de6f57 ||
           shader_hash == 0x57bd9f27;
}

[[nodiscard]] constexpr bool ComputeDispatchRequiresIdleBeforeChunks(u64 shader_hash) noexcept {
    return shader_hash == 0x87cf9b8a;
}

[[nodiscard]] constexpr u32 ComputeDispatchMaxChunkX(u64 shader_hash, u32 dim_x) noexcept {
    // These shaders have repeatedly exceeded the Windows NVIDIA TDR budget as single dispatches.
    if (shader_hash == 0x87cf9b8a || shader_hash == 0x4defcf7f) {
        return 1;
    }
    if (shader_hash == 0xbf4600a0) {
        return dim_x <= 128 ? 1 : ~u32{0};
    }
    if (shader_hash == 0x50784801 && dim_x > 1024) {
        return 128;
    }
    if (shader_hash == 0x19de6f57 && dim_x > 128) {
        return 128;
    }
    if (shader_hash == 0x57bd9f27 && dim_x > 16) {
        return 16;
    }
    return ~u32{0};
}

[[nodiscard]] constexpr u32 ComputeDispatchMaxChunkY(u64 shader_hash, u32 dim_y) noexcept {
    if (shader_hash == 0x87cf9b8a) {
        return 1;
    }
    if ((shader_hash == 0x950e34f8 || shader_hash == 0x50784801) && dim_y > 16) {
        return 16;
    }
    if (shader_hash == 0x57bd9f27 && dim_y > 8) {
        return 8;
    }
    return ~u32{0};
}

struct ComputeDispatchChunk {
    u32 base_x{};
    u32 base_y{};
    u32 group_count_x{};
    u32 group_count_y{1};

    constexpr bool operator==(const ComputeDispatchChunk&) const noexcept = default;
};

[[nodiscard]] constexpr bool ComputeDispatchChunkNeedsBase(
    const ComputeDispatchChunk& chunk) noexcept {
    return chunk.base_x != 0 || chunk.base_y != 0;
}

[[nodiscard]] constexpr u32 ComputeDispatchChunkCount(u32 dim_x, u32 max_chunk_x) noexcept {
    if (dim_x == 0 || max_chunk_x == 0) {
        return 0;
    }
    return 1 + (dim_x - 1) / max_chunk_x;
}

[[nodiscard]] constexpr ComputeDispatchChunk ComputeDispatchChunkAt(
    u32 dim_x, u32 max_chunk_x, u32 chunk_index) noexcept {
    if (dim_x == 0 || max_chunk_x == 0) {
        return {};
    }
    const u32 base_x = chunk_index * max_chunk_x;
    if (base_x >= dim_x) {
        return {};
    }
    return {.base_x = base_x,
            .base_y = 0,
            .group_count_x = std::min(max_chunk_x, dim_x - base_x),
            .group_count_y = 1};
}

[[nodiscard]] constexpr u32 ComputeDispatchChunk2DCount(u32 dim_x, u32 dim_y,
                                                        u32 max_chunk_x,
                                                        u32 max_chunk_y) noexcept {
    return ComputeDispatchChunkCount(dim_x, max_chunk_x) *
           ComputeDispatchChunkCount(dim_y, max_chunk_y);
}

[[nodiscard]] constexpr ComputeDispatchChunk ComputeDispatchChunk2DAt(
    u32 dim_x, u32 dim_y, u32 max_chunk_x, u32 max_chunk_y, u32 chunk_index) noexcept {
    const u32 chunks_x = ComputeDispatchChunkCount(dim_x, max_chunk_x);
    const u32 chunks_y = ComputeDispatchChunkCount(dim_y, max_chunk_y);
    if (chunks_x == 0 || chunks_y == 0 || chunk_index >= chunks_x * chunks_y) {
        return {};
    }
    const u32 chunk_x = chunk_index % chunks_x;
    const u32 chunk_y = chunk_index / chunks_x;
    const u32 base_x = chunk_x * max_chunk_x;
    const u32 base_y = chunk_y * max_chunk_y;
    return {.base_x = base_x,
            .base_y = base_y,
            .group_count_x = std::min(max_chunk_x, dim_x - base_x),
            .group_count_y = std::min(max_chunk_y, dim_y - base_y)};
}

} // namespace Vulkan
