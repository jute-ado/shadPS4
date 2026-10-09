// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <limits>
#include <utility>
#include "common/types.h"

namespace VideoCore {

enum class ImageBindingKind {
    OneDimensional,
    TwoDimensional,
    ThreeDimensional,
    Cube,
};

struct ImageBindingExtent {
    u32 width{};
    u32 height{};
    u32 depth{};
    u32 layers{};
    u32 levels{1};
};

struct ImageBindingLimits {
    u32 max_dimension_1d{};
    u32 max_dimension_2d{};
    u32 max_dimension_3d{};
    u32 max_dimension_cube{};
    u32 max_array_layers{};
    u64 max_guest_resource_size{std::numeric_limits<u64>::max()};
};

[[nodiscard]] constexpr bool IsImageBindingExtentUsable(const ImageBindingKind kind,
                                                        const ImageBindingExtent extent,
                                                        const ImageBindingLimits limits) {
    if (extent.width == 0 || extent.height == 0 || extent.depth == 0 || extent.layers == 0 ||
        extent.levels == 0 || extent.layers > limits.max_array_layers ||
        extent.levels > std::bit_width(std::max({extent.width, extent.height, extent.depth}))) {
        return false;
    }

    switch (kind) {
    case ImageBindingKind::OneDimensional:
        return extent.width <= limits.max_dimension_1d && extent.height == 1 && extent.depth == 1;
    case ImageBindingKind::TwoDimensional:
        return extent.width <= limits.max_dimension_2d && extent.height <= limits.max_dimension_2d &&
               extent.depth == 1;
    case ImageBindingKind::ThreeDimensional:
        return extent.width <= limits.max_dimension_3d && extent.height <= limits.max_dimension_3d &&
               extent.depth <= limits.max_dimension_3d && extent.layers == 1;
    case ImageBindingKind::Cube:
        return extent.width <= limits.max_dimension_cube && extent.height == extent.width &&
               extent.depth == 1;
    }
    return false;
}

template <typename IsMapped>
[[nodiscard]] bool IsGuestImageSpecializationUsable(
    const VAddr address, const u32 encoded_pitch, const u32 encoded_width,
    const ImageBindingKind kind, const ImageBindingExtent extent, const ImageBindingLimits limits,
    IsMapped&& is_mapped) {
    return address != 0 && encoded_pitch >= encoded_width &&
           std::forward<IsMapped>(is_mapped)(address, 1) &&
           IsImageBindingExtentUsable(kind, extent, limits);
}

template <typename IsMapped>
[[nodiscard]] bool IsGuestImageBindingUsable(const VAddr address, const u32 encoded_pitch,
                                             const u32 encoded_width, const ImageBindingKind kind,
                                             const ImageBindingExtent extent,
                                             const u64 required_size,
                                             const ImageBindingLimits limits,
                                             IsMapped&& is_mapped) {
    return required_size != 0 && required_size <= limits.max_guest_resource_size &&
           address <= std::numeric_limits<VAddr>::max() - required_size &&
           IsGuestImageSpecializationUsable(address, encoded_pitch, encoded_width, kind, extent,
                                             limits, std::forward<IsMapped>(is_mapped));
}

} // namespace VideoCore
