// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "common/types.h"
#include "video_core/amdgpu/resource.h"

namespace VideoCore {

struct SupportedImageUsage {
    vk::ImageUsageFlags usage{};
    bool dropped_optional_storage{};
};

struct HostImageResourceExtent {
    u32 width{};
    u32 height{};
    u32 depth{};
};

struct HostImageSubresourceExtent {
    u32 levels{};
    u32 layers{};
};

struct HostImageResourceLimits {
    u32 max_1d_dimension{};
    u32 max_2d_dimension{};
    u32 max_3d_dimension{};
    u32 max_array_layers{};
    u32 max_mip_levels{};
};

[[nodiscard]] constexpr bool ShouldAllowDiagnosticIncompleteNonSparseImage(
    const bool has_lod_feedback, const bool is_written, const bool diagnostic_enabled) {
    return diagnostic_enabled && has_lod_feedback && !is_written;
}

class LodFeedbackImageTracker {
public:
    void Observe(const VAddr address, const u32 size, const bool has_lod_feedback) {
        if (!has_lod_feedback || size == 0) {
            return;
        }
        auto& observed_size = allocations[address];
        observed_size = std::max(observed_size, size);
    }

    [[nodiscard]] bool CanUseIncompleteBacking(const VAddr address, const u32 size,
                                               const bool is_written,
                                               const bool diagnostic_enabled) const {
        if (!ShouldAllowDiagnosticIncompleteNonSparseImage(true, is_written,
                                                           diagnostic_enabled) ||
            size == 0) {
            return false;
        }
        const auto allocation = allocations.find(address);
        return allocation != allocations.end() && size <= allocation->second;
    }

    void Invalidate(const VAddr address, const u64 size) {
        if (size == 0) {
            return;
        }
        const VAddr end = address + std::min(size, std::numeric_limits<VAddr>::max() - address);
        for (auto allocation = allocations.begin(); allocation != allocations.end();) {
            const VAddr allocation_end =
                allocation->first +
                std::min<u64>(allocation->second,
                              std::numeric_limits<VAddr>::max() - allocation->first);
            if (allocation->first < end && address < allocation_end) {
                allocation = allocations.erase(allocation);
            } else {
                ++allocation;
            }
        }
    }

private:
    std::unordered_map<VAddr, u32> allocations;
};

[[nodiscard]] constexpr bool IsImageGuestBackingUsable(const AmdGpu::ArrayMode array_mode,
                                                       const bool fully_mapped,
                                                       const bool allow_incomplete_non_sparse = false) {
    if (allow_incomplete_non_sparse) {
        return true;
    }
    using enum AmdGpu::ArrayMode;
    switch (array_mode) {
    case ArrayPrtTiledThin1:
    case ArrayPrt2DTiledThin1:
    case ArrayPrt3DTiledThin1:
    case ArrayPrtTiledThick:
    case ArrayPrt2DTiledThick:
    case ArrayPrt3DTiledThick:
        return true;
    default:
        return fully_mapped;
    }
}

[[nodiscard]] constexpr bool IsImageEstimatedAllocationWithinLimit(
    HostImageResourceExtent extent, const HostImageSubresourceExtent resources,
    const u32 bits_per_block, const bool is_block_encoded, const u32 samples,
    const u64 byte_limit) {
    if (extent.width == 0 || extent.height == 0 || extent.depth == 0 || resources.layers == 0 ||
        bits_per_block == 0 || samples == 0 || byte_limit == 0) {
        return false;
    }
    if (is_block_encoded) {
        extent.width = (extent.width + 3) / 4;
        extent.height = (extent.height + 3) / 4;
    }
    const std::array<u64, 6> factors{
        extent.width, extent.height, extent.depth, resources.layers, samples,
        (bits_per_block + 7U) / 8U,
    };
    u64 bytes = 1;
    for (const u64 factor : factors) {
        if (bytes > byte_limit / factor) {
            return false;
        }
        bytes *= factor;
    }
    return true;
}

template <typename SupportsFormat>
[[nodiscard]] constexpr bool IsImageDescriptorFormatUsable(
    const AmdGpu::DataFormat data_format, const AmdGpu::NumberFormat number_format,
    SupportsFormat&& supports_format) {
    if (data_format == AmdGpu::DataFormat::FormatInvalid) {
        return false;
    }
    return std::forward<SupportsFormat>(supports_format)(data_format, number_format);
}

[[nodiscard]] constexpr bool IsImageComponentMappingUsable(const u32 red, const u32 green,
                                                           const u32 blue, const u32 alpha) {
    const auto usable = [](const u32 component) {
        return component == 0 || component == 1 || (component >= 4 && component <= 7);
    };
    return usable(red) && usable(green) && usable(blue) && usable(alpha);
}

[[nodiscard]] constexpr bool IsImageResourceWithinHostLimits(
    const AmdGpu::ImageType type, const HostImageResourceExtent extent,
    const HostImageSubresourceExtent resources, const HostImageResourceLimits limits) {
    if (extent.width == 0 || resources.levels == 0 || resources.layers == 0 ||
        resources.levels > limits.max_mip_levels) {
        return false;
    }

    switch (type) {
    case AmdGpu::ImageType::Color1D:
        return extent.width <= limits.max_1d_dimension && resources.layers == 1;
    case AmdGpu::ImageType::Color1DArray:
        return extent.width <= limits.max_1d_dimension &&
               resources.layers <= limits.max_array_layers;
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
        return extent.height != 0 && extent.width <= limits.max_2d_dimension &&
               extent.height <= limits.max_2d_dimension && resources.layers == 1;
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Color2DMsaaArray:
        return extent.height != 0 && extent.width <= limits.max_2d_dimension &&
               extent.height <= limits.max_2d_dimension &&
               resources.layers <= limits.max_array_layers;
    case AmdGpu::ImageType::Color3D:
        return extent.height != 0 && extent.depth != 0 &&
               extent.width <= limits.max_3d_dimension &&
               extent.height <= limits.max_3d_dimension &&
               extent.depth <= limits.max_3d_dimension && resources.layers == 1;
    default:
        return false;
    }
}

template <typename ImageInfoType, typename ResourceExtent>
void ExpandImageResources(ImageInfoType& info, const ResourceExtent requested) {
    const ResourceExtent expanded = std::max(info.resources, requested);
    if (expanded == info.resources) {
        return;
    }
    info.resources = expanded;
    info.UpdateSize();
}

[[nodiscard]] constexpr vk::ImageUsageFlags WithAttachmentFeedbackLoopUsage(
    vk::ImageUsageFlags usage, const bool supported) {
    return supported ? usage | vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT : usage;
}

[[nodiscard]] constexpr vk::ImageAspectFlags AttachmentFeedbackLoopAspects(
    const bool is_depth, const bool has_stencil) {
    if (!is_depth) {
        return vk::ImageAspectFlagBits::eColor;
    }
    vk::ImageAspectFlags aspects = vk::ImageAspectFlagBits::eDepth;
    if (has_stencil) {
        aspects |= vk::ImageAspectFlagBits::eStencil;
    }
    return aspects;
}

template <typename SupportsUsage>
[[nodiscard]] std::optional<SupportedImageUsage> SelectSupportedImageUsage(
    const bool is_block_encoded, const vk::ImageUsageFlags preferred_usage,
    SupportsUsage&& supports_usage) {
    if (std::forward<SupportsUsage>(supports_usage)(preferred_usage)) {
        return SupportedImageUsage{.usage = preferred_usage};
    }

    if (!is_block_encoded || !(preferred_usage & vk::ImageUsageFlagBits::eStorage)) {
        return std::nullopt;
    }

    const auto required_usage = preferred_usage & ~vk::ImageUsageFlagBits::eStorage;
    if (!std::forward<SupportsUsage>(supports_usage)(required_usage)) {
        return std::nullopt;
    }
    return SupportedImageUsage{
        .usage = required_usage,
        .dropped_optional_storage = true,
    };
}

} // namespace VideoCore
