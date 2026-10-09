// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <optional>
#include <span>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace VideoCore {

enum class DepthCopyMode { Direct, Buffer, Unsupported };

/// Vulkan buffer/image copies represent depth and stencil as separate planes.
[[nodiscard]] constexpr vk::Format DepthPlaneFormat(vk::Format format) {
    switch (format) {
    case vk::Format::eD16Unorm:
    case vk::Format::eD16UnormS8Uint:
        return vk::Format::eD16Unorm;
    case vk::Format::eX8D24UnormPack32:
    case vk::Format::eD24UnormS8Uint:
        return vk::Format::eX8D24UnormPack32;
    case vk::Format::eD32Sfloat:
    case vk::Format::eD32SfloatS8Uint:
        return vk::Format::eD32Sfloat;
    default:
        return vk::Format::eUndefined;
    }
}

/// maintenance8 does not relax the identical-format rule between two depth images.
/// This policy leaves identical formats and existing color/depth handling unchanged.
[[nodiscard]] constexpr DepthCopyMode SelectDepthCopyMode(vk::Format source, vk::Format destination,
                                                          unsigned source_samples,
                                                          unsigned destination_samples) {
    const auto source_plane = DepthPlaneFormat(source);
    const auto destination_plane = DepthPlaneFormat(destination);
    if (source == destination || source_plane == vk::Format::eUndefined ||
        destination_plane == vk::Format::eUndefined) {
        return DepthCopyMode::Direct;
    }
    if (source_plane != destination_plane || source_samples != 1 || destination_samples != 1) {
        return DepthCopyMode::Unsupported;
    }
    return DepthCopyMode::Buffer;
}

struct DepthCopyBufferPlan {
    std::vector<vk::BufferImageCopy> sources;
    std::vector<vk::BufferImageCopy> destinations;
    vk::DeviceSize byte_size{};
};

/// Builds tightly packed, non-overlapping depth-plane ranges; never numeric conversion.
[[nodiscard]] inline std::optional<DepthCopyBufferPlan> BuildDepthCopyBufferPlan(
    vk::Format source, vk::Format destination, std::span<const vk::ImageCopy> copies) {
    const auto plane = DepthPlaneFormat(source);
    if (plane == vk::Format::eUndefined || plane != DepthPlaneFormat(destination) ||
        copies.empty()) {
        return std::nullopt;
    }

    constexpr auto MaxSize = std::numeric_limits<vk::DeviceSize>::max();
    const vk::DeviceSize texel_bytes = plane == vk::Format::eD16Unorm ? 2 : 4;
    DepthCopyBufferPlan plan;
    plan.sources.reserve(copies.size());
    plan.destinations.reserve(copies.size());
    for (const auto& copy : copies) {
        if (!(copy.srcSubresource.aspectMask & vk::ImageAspectFlagBits::eDepth) ||
            !(copy.dstSubresource.aspectMask & vk::ImageAspectFlagBits::eDepth) ||
            copy.srcSubresource.layerCount == 0 ||
            copy.srcSubresource.layerCount != copy.dstSubresource.layerCount ||
            copy.srcOffset.x < 0 || copy.srcOffset.y < 0 || copy.srcOffset.z < 0 ||
            copy.dstOffset.x < 0 || copy.dstOffset.y < 0 || copy.dstOffset.z < 0) {
            return std::nullopt;
        }
        vk::DeviceSize size = texel_bytes;
        for (const auto factor : {copy.extent.width, copy.extent.height, copy.extent.depth,
                                  copy.srcSubresource.layerCount}) {
            if (factor == 0 || size > MaxSize / factor) {
                return std::nullopt;
            }
            size *= factor;
        }
        // A four-byte-aligned offset satisfies both depth-plane formats and all queue types.
        if (plan.byte_size > MaxSize - 3) {
            return std::nullopt;
        }
        const vk::DeviceSize offset = (plan.byte_size + 3) & ~vk::DeviceSize{3};
        if (size > MaxSize - offset) {
            return std::nullopt;
        }

        vk::BufferImageCopy source_copy{};
        source_copy.bufferOffset = offset;
        source_copy.imageSubresource = copy.srcSubresource;
        source_copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
        source_copy.imageOffset = copy.srcOffset;
        source_copy.imageExtent = copy.extent;
        auto destination_copy = source_copy;
        destination_copy.imageSubresource = copy.dstSubresource;
        destination_copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
        destination_copy.imageOffset = copy.dstOffset;
        plan.sources.push_back(source_copy);
        plan.destinations.push_back(destination_copy);
        plan.byte_size = offset + size;
    }
    return plan;
}

} // namespace VideoCore
