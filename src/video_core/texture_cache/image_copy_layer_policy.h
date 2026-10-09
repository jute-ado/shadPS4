// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>

#include "common/types.h"

namespace VideoCore {

/// Layer counts for a copy between images with the same Vulkan image type.
struct CopyLayerCounts {
    u32 source;
    u32 destination;

    auto operator<=>(const CopyLayerCounts&) const = default;
};

/// Bound a same-type image copy to the layers exposed by both source and destination views.
///
/// Vulkan requires the source and destination layer counts in VkImageCopy/VkImageResolve to match.
/// A view can expose fewer layers than its backing image (for example, a non-zero base slice), so
/// the full image resource counts are not sufficient when constructing a copy region.
[[nodiscard]] constexpr CopyLayerCounts ClampEqualImageCopyLayersToRanges(
    const CopyLayerCounts image_layers, const u32 source_range_layers,
    const u32 destination_range_layers) noexcept {
    const u32 copy_layers = std::min({image_layers.source, image_layers.destination,
                                      source_range_layers, destination_range_layers});
    return {copy_layers, copy_layers};
}

} // namespace VideoCore
