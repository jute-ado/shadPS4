// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>

#include "common/types.h"
#include "video_core/amdgpu/tiling.h"

namespace VideoCore {

// TILING_COMP is compiled with BITS_PER_PIXEL and NUM_SAMPLES defines.  Keep every supported
// specialization in a distinct cache slot; using std::bit_width for bpp would alias 64 and 96,
// while a key without samples would reuse the wrong address formula for MSAA surfaces.
inline constexpr size_t TilingPipelineBitsPerPixelVariants = 6;
inline constexpr size_t TilingPipelineSampleVariants = 5;
inline constexpr size_t TilingPipelineCacheSize =
    AmdGpu::NUM_TILE_MODES * TilingPipelineBitsPerPixelVariants * TilingPipelineSampleVariants;

[[nodiscard]] constexpr std::optional<size_t> TilingPipelineBitsPerPixelIndex(
    const u32 bits_per_pixel) noexcept {
    switch (bits_per_pixel) {
    case 8:
        return 0;
    case 16:
        return 1;
    case 32:
        return 2;
    case 64:
        return 3;
    case 96:
        return 4;
    case 128:
        return 5;
    default:
        return std::nullopt;
    }
}

[[nodiscard]] constexpr std::optional<size_t> TilingPipelineSampleIndex(
    const u32 num_samples) noexcept {
    switch (num_samples) {
    case 1:
        return 0;
    case 2:
        return 1;
    case 4:
        return 2;
    case 8:
        return 3;
    case 16:
        return 4;
    default:
        return std::nullopt;
    }
}

[[nodiscard]] constexpr std::optional<size_t> TilingPipelineCacheIndex(
    const AmdGpu::TileMode tile_mode, const u32 bits_per_pixel, const u32 num_samples) noexcept {
    const auto tile_index = static_cast<size_t>(tile_mode);
    const auto bpp_index = TilingPipelineBitsPerPixelIndex(bits_per_pixel);
    const auto sample_index = TilingPipelineSampleIndex(num_samples);
    // Encoding 30 is reserved/unknown in the current GFX8 table.  Do not let it acquire a cache
    // slot merely because it is numerically inside the five-bit range.
    if (tile_index >= AmdGpu::NUM_TILE_MODES || tile_index == 30 || !bpp_index || !sample_index) {
        return std::nullopt;
    }
    return (tile_index * TilingPipelineBitsPerPixelVariants + *bpp_index) *
               TilingPipelineSampleVariants +
           *sample_index;
}

} // namespace VideoCore
