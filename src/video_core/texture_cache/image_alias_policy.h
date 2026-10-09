// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>

#include "video_core/amdgpu/resource.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_write_history.h"

namespace VideoCore {

enum class ImageAliasDisposition {
    ExistingRules,
    Compatible,
    Separate,
    RetireClean,
    Unsupported,
    InvalidLayout,
};

// A cache image can only be reused as a view when the guest byte layout of every
// mip shared by the two images is identical.  A longer request with an identical
// cached prefix is intentionally compatible so ResolveOverlap can preserve the
// existing ExpandImage path. Total guest size and BlockDim() are insufficient for
// block-compressed surfaces: pitches are rounded to block/tile boundaries and a
// one-byte encoded pitch change can move all following mip offsets.
enum class ImageLayoutCompatibility {
    Unknown,
    Compatible,
    Incompatible,
};

[[nodiscard]] inline bool HasValidPhysicalImageLayout(const ImageInfo& info) {
    if (info.resources.levels == 0 || info.resources.levels > info.mips_layout.size() ||
        info.resources.layers == 0 || info.num_bits == 0 || info.num_samples == 0) {
        return false;
    }
    for (u32 mip = 0; mip < info.resources.levels; ++mip) {
        const auto& layout = info.mips_layout[mip];
        if (layout.size == 0 || layout.pitch == 0 || layout.height == 0 ||
            layout.offset > info.guest_size || layout.size > info.guest_size - layout.offset) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] inline ImageLayoutCompatibility ComparePhysicalImageLayout(
    const ImageInfo& requested, const ImageInfo& cached) {
    // Color/depth buffers and other legacy descriptors may not populate mip metadata.  Leave
    // those paths under the existing overlap rules rather than turning missing metadata into a
    // new rejection.
    if (!HasValidPhysicalImageLayout(requested) || !HasValidPhysicalImageLayout(cached)) {
        return ImageLayoutCompatibility::Unknown;
    }
    const bool same_surface_layout =
        requested.resources.layers == cached.resources.layers &&
        requested.props.is_block == cached.props.is_block &&
        requested.props.is_tiled == cached.props.is_tiled &&
        requested.num_bits == cached.num_bits && requested.num_samples == cached.num_samples &&
        requested.tile_mode == cached.tile_mode && requested.array_mode == cached.array_mode &&
        requested.bank_swizzle == cached.bank_swizzle && requested.alt_tile == cached.alt_tile;
    if (!same_surface_layout) {
        return ImageLayoutCompatibility::Incompatible;
    }

    // A cached image may contain only a mip prefix of the requested image.  When that
    // prefix has the same physical layout, ResolveOverlap must be allowed to reach its
    // existing ExpandImage path so the already materialized mips are copied forward.
    // Compare only the levels present in both images; a mismatch in that common prefix
    // remains incompatible in either direction.
    const u32 common_levels = std::min(requested.resources.levels, cached.resources.levels);
    for (u32 mip = 0; mip < common_levels; ++mip) {
        const auto& lhs = requested.mips_layout[mip];
        const auto& rhs = cached.mips_layout[mip];
        if (lhs.size != rhs.size || lhs.pitch != rhs.pitch || lhs.height != rhs.height ||
            lhs.offset != rhs.offset) {
            return ImageLayoutCompatibility::Incompatible;
        }
    }
    return ImageLayoutCompatibility::Compatible;
}

[[nodiscard]] inline bool CanConsiderPerfectImageAlias(ImageAliasDisposition disposition) {
    return disposition == ImageAliasDisposition::ExistingRules ||
           disposition == ImageAliasDisposition::Compatible;
}

class ImageAliasDebugBudget {
public:
    bool Take(bool enabled) {
        if (!enabled || remaining == 0) {
            return false;
        }
        --remaining;
        return true;
    }

private:
    u32 remaining = 16;
};

// These ordinary same-shape aliases can be compared without changing the existing explicit
// depth, compressed/uncompressed, volume, or array-layer reinterpretation paths. A mip prefix
// is compatible only when its actual guest byte layout agrees, not merely its total size.
[[nodiscard]] inline ImageAliasDisposition PlanSameBaseImageAlias(const ImageInfo& requested,
                                                                  const ImageInfo& cached,
                                                                  const ImageWriteHistory& history,
                                                                  bool gpu_modified,
                                                                  bool old_enough) {
    if (requested.guest_address != cached.guest_address || requested.props.is_depth ||
        cached.props.is_depth || requested.props.is_block != cached.props.is_block ||
        requested.pixel_format != cached.pixel_format || requested.type != cached.type ||
        requested.type == AmdGpu::ImageType::Color3D || requested.size != cached.size ||
        requested.resources.layers != cached.resources.layers) {
        return ImageAliasDisposition::ExistingRules;
    }

    const auto has_valid_layout = [](const ImageInfo& info) {
        if (info.resources.levels == 0 || info.resources.levels > info.mips_layout.size() ||
            info.resources.layers == 0 || info.num_bits == 0 || info.num_samples == 0) {
            return false;
        }
        for (u32 mip = 0; mip < info.resources.levels; ++mip) {
            const auto& layout = info.mips_layout[mip];
            if (layout.size == 0 || layout.pitch == 0 || layout.height == 0 ||
                layout.offset > info.guest_size || layout.size > info.guest_size - layout.offset) {
                return false;
            }
        }
        return true;
    };
    if (!has_valid_layout(requested) || !has_valid_layout(cached)) {
        return ImageAliasDisposition::InvalidLayout;
    }

    bool same_layout =
        requested.num_bits == cached.num_bits && requested.num_samples == cached.num_samples &&
        requested.tile_mode == cached.tile_mode && requested.array_mode == cached.array_mode &&
        requested.props.is_tiled == cached.props.is_tiled &&
        requested.bank_swizzle == cached.bank_swizzle && requested.alt_tile == cached.alt_tile;
    const u32 common_levels = std::min(requested.resources.levels, cached.resources.levels);
    for (u32 mip = 0; same_layout && mip < common_levels; ++mip) {
        const auto& lhs = requested.mips_layout[mip];
        const auto& rhs = cached.mips_layout[mip];
        same_layout = lhs.size == rhs.size && lhs.pitch == rhs.pitch && lhs.height == rhs.height &&
                      lhs.offset == rhs.offset;
    }
    if (same_layout) {
        return ImageAliasDisposition::Compatible;
    }

    // A different layout needs independent guest-byte materialization. Neither age nor the
    // refresh Dirty flags prove that dropping an image with image-only writes is lossless.
    // Such aliases need a real old-layout tile -> guest buffer -> new-layout detile path.
    if (gpu_modified || history.MayHaveImageOnlyContents()) {
        return ImageAliasDisposition::Unsupported;
    }
    return old_enough ? ImageAliasDisposition::RetireClean : ImageAliasDisposition::Separate;
}

} // namespace VideoCore
