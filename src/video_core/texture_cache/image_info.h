// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <limits>

#include "common/types.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/types.h"

namespace AmdGpu {
struct ColorBuffer;
struct DepthBuffer;
struct Image;
enum class ImageType : u64;
} // namespace AmdGpu

namespace Libraries::VideoOut {
struct BufferAttributeGroup;
}

namespace Shader {
struct ImageResource;
}

namespace VideoCore {

/// Whether a guest image's array-layer count can be represented by the host image queried for its
/// format and usage.  Keep this as a pure admission predicate so malformed descriptors are
/// rejected before VkImageCreateInfo or any buffer-image copy can be published.
[[nodiscard]] constexpr bool IsHostImageArrayLayerCountSupported(
    const u32 requested_layers, const u32 max_array_layers) noexcept {
    return requested_layers != 0 && max_array_layers != 0 && requested_layers <= max_array_layers;
}

/// Whether a guest image's extent and mip count can be represented by the host image format
/// query.  These are exact Vulkan capability limits and must be checked before VkImageCreateInfo
/// is submitted; unlike maxResourceSize, they do not require guessing the driver's private
/// optimal-tiling allocation footprint from the guest layout.
[[nodiscard]] constexpr bool IsHostImageGeometrySupported(
    const Extent3D requested_extent, const u32 requested_levels, const Extent3D max_extent,
    const u32 max_mip_levels) noexcept {
    return requested_extent.width != 0 && requested_extent.height != 0 &&
           requested_extent.depth != 0 && requested_levels != 0 && max_extent.width != 0 &&
           max_extent.height != 0 && max_extent.depth != 0 && max_mip_levels != 0 &&
           requested_extent.width <= max_extent.width &&
           requested_extent.height <= max_extent.height &&
           requested_extent.depth <= max_extent.depth && requested_levels <= max_mip_levels;
}

/// Guest image ranges use a 32-bit byte size.  Keep the check separate from host Vulkan limits so
/// layout derivation can fail closed before a wrapped range reaches tracking, tiling, or copies.
[[nodiscard]] constexpr bool IsGuestImageFootprintRepresentable(const u64 footprint) noexcept {
    return footprint != 0 && footprint <= std::numeric_limits<u32>::max();
}

struct ImageProperties {
    u32 is_volume : 1;
    u32 is_tiled : 1;
    u32 is_pow2 : 1;
    u32 is_block : 1;
    u32 is_depth : 1;
    u32 has_stencil : 1;
};

struct ImageInfo {
    ImageInfo() = default;
    ImageInfo(const Libraries::VideoOut::BufferAttributeGroup& group, VAddr cpu_address) noexcept;
    ImageInfo(const AmdGpu::ColorBuffer& buffer, AmdGpu::CbDbExtent hint) noexcept;
    ImageInfo(const AmdGpu::DepthBuffer& buffer, u32 num_slices, VAddr htile_address,
              AmdGpu::CbDbExtent hint, bool write_buffer = false) noexcept;
    ImageInfo(const AmdGpu::Image& image, const Shader::ImageResource& desc) noexcept;

    /// Whether the descriptor's five-bit tile-mode encoding is known to this implementation.
    /// Unknown/reserved encodings must be rejected before deriving guest layout or creating a
    /// Vulkan image; they are not interchangeable with either linear layout.
    [[nodiscard]] bool HasValidTileMode() const noexcept {
        return AmdGpu::TryGetArrayMode(tile_mode).has_value();
    }

    /// Vulkan only permits the complete mip chain implied by the image extent. Guest descriptors
    /// can carry stale last-level bits after a surface is recycled, so reject those chains before
    /// constructing an image rather than relying on validation-layer diagnostics at vkCreateImage.
    [[nodiscard]] bool HasValidMipChain() const noexcept {
        if (!has_valid_guest_size || resources.levels == 0 ||
            resources.levels > mips_layout.size() || resources.layers == 0 ||
            size.width == 0 || size.height == 0 || size.depth == 0 || guest_size == 0) {
            return false;
        }
        // A valid extent alone is not enough for admission.  Keep the layout table and guest
        // footprint transactional: every mip must be present exactly once, in offset order, and
        // the final byte must be the end of guest_size.  This also rejects partially initialized
        // ImageInfo values before a host image can be published.
        u64 expected_offset = 0;
        for (u32 mip = 0; mip < resources.levels; ++mip) {
            const auto& layout = mips_layout[mip];
            if (layout.size == 0 || layout.pitch == 0 || layout.height == 0 ||
                layout.offset != expected_offset ||
                expected_offset + layout.size > guest_size) {
                return false;
            }
            expected_offset += layout.size;
        }
        if (expected_offset != guest_size) {
            return false;
        }
        u32 largest_dimension = std::max(size.width, size.height);
        if (props.is_volume) {
            largest_dimension = std::max(largest_dimension, size.depth);
        }
        u32 complete_levels = 1;
        while (largest_dimension > 1) {
            largest_dimension >>= 1;
            ++complete_levels;
        }
        return resources.levels <= complete_levels;
    }

    bool IsTiled() const {
        return HasValidTileMode() && tile_mode != AmdGpu::TileMode::DisplayLinearAligned &&
               tile_mode != AmdGpu::TileMode::DisplayLinearGeneral;
    }

    Extent2D BlockDim() const {
        const auto dim = props.is_block ? 2 : 0;
        return Extent2D{pitch >> dim, size.height >> dim};
    }

    s32 MipOf(const ImageInfo& info) const;
    s32 SliceOf(const ImageInfo& info, s32 mip) const;

    bool IsCompatible(const ImageInfo& info) const;
    void UpdateSize();

    struct {
        VAddr cmask_addr;
        VAddr fmask_addr;
        VAddr htile_addr;
        s32 htile_clear_mask = -1;
    } meta_info{};

    ImageProperties props{};
    vk::Format pixel_format = vk::Format::eUndefined;
    AmdGpu::ImageType type{};
    SubresourceExtent resources;
    Extent3D size{1, 1, 1};
    u32 num_bits{};
    u32 num_samples = 1;
    u32 pitch{};
    AmdGpu::TileMode tile_mode = AmdGpu::TileMode::DisplayLinearAligned;
    AmdGpu::ArrayMode array_mode = AmdGpu::ArrayMode::ArrayLinearAligned;
    struct MipInfo {
        u32 size;
        u32 pitch;
        u32 height;
        u32 offset;
    };
    std::array<MipInfo, 16> mips_layout{};
    VAddr guest_address{};
    u32 guest_size{};
    bool has_valid_guest_size = true;
    // Shader image writes require storage usage on the host image.  Read-only descriptors may
    // drop the cache's optional storage usage when multisampled storage is unsupported.
    bool storage_required{};
    u8 bank_swizzle{};
    bool alt_tile{};

    VAddr stencil_addr{};
    u32 stencil_size{};
};

} // namespace VideoCore
