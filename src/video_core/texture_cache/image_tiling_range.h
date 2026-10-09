// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <optional>
#include <span>

#include "video_core/texture_cache/image_buffer_transfer_policy.h"
#include "video_core/texture_cache/image_info.h"

namespace VideoCore {

struct ImageTilingPlan {
    u32 byte_size;
    u32 element_count;
    u32 dispatch_x;
    u32 num_slices;
    u32 num_mips;
    std::array<ImageInfo::MipInfo, 16> mips{};
};

// TileImage copies a consecutive prefix of complete mips, including all array layers or
// volume slices. Both descriptor ranges and shader work must describe that same prefix,
// not the complete image which may extend beyond the destination buffer.
[[nodiscard]] inline std::optional<ImageTilingPlan> PlanImageTilingRange(
    const ImageInfo& info, std::span<const vk::BufferImageCopy> copies, u64 output_offset,
    u64 output_size) {
    if (copies.empty() || info.resources.levels == 0 ||
        info.resources.levels > info.mips_layout.size() || copies.size() > info.resources.levels ||
        info.num_bits < 8 || info.num_bits % 8 != 0 ||
        (info.props.is_tiled &&
         (info.num_bits > 128 || (!std::has_single_bit(info.num_bits) && info.num_bits != 96))) ||
        info.size.width == 0 || info.size.height == 0 || info.size.depth == 0 ||
        info.resources.layers == 0 || !SupportsImageBufferTransfer(info.num_samples) ||
        output_offset > output_size) {
        return std::nullopt;
    }
    const u32 element_bytes = info.num_bits / 8;
    ImageTilingPlan plan{
        .byte_size = 0,
        .element_count = 0,
        .dispatch_x = 0,
        .num_slices = info.props.is_volume ? info.size.depth : info.resources.layers,
        .num_mips = static_cast<u32>(copies.size()),
    };
    for (u32 mip = 0; mip < plan.num_mips; ++mip) {
        const auto& layout = info.mips_layout[mip];
        const auto& copy = copies[mip];
        const u32 width = std::max(info.size.width >> mip, 1U);
        const u32 height = std::max(info.size.height >> mip, 1U);
        const u32 depth = std::max(info.size.depth >> mip, 1U);
        if (layout.offset != plan.byte_size || layout.size == 0 ||
            layout.size % element_bytes != 0 || layout.pitch < width || layout.height < height ||
            layout.size > info.guest_size - plan.byte_size || copy.bufferOffset != layout.offset ||
            copy.bufferRowLength != layout.pitch || copy.bufferImageHeight != layout.height ||
            copy.imageSubresource.mipLevel != mip || copy.imageSubresource.baseArrayLayer != 0 ||
            copy.imageSubresource.layerCount != info.resources.layers ||
            copy.imageOffset != vk::Offset3D{0, 0, 0} ||
            copy.imageExtent != vk::Extent3D{width, height, depth}) {
            return std::nullopt;
        }
        plan.mips[mip] = info.mips_layout[mip];
        if (info.props.is_block) {
            plan.mips[mip].pitch = static_cast<u32>((u64{layout.pitch} + 3) / 4);
            plan.mips[mip].height = static_cast<u32>((u64{layout.height} + 3) / 4);
        }
        // Reject impossible geometry without overflowing a pitch * height * slices product.
        u64 required_bytes = element_bytes;
        for (const u32 factor : {plan.mips[mip].pitch, plan.mips[mip].height, depth,
                                 info.resources.layers, info.num_samples}) {
            if (factor > layout.size / required_bytes) {
                return std::nullopt;
            }
            required_bytes *= factor;
        }
        plan.byte_size += layout.size;
    }
    if (plan.byte_size > output_size - output_offset) {
        return std::nullopt;
    }
    plan.element_count = plan.byte_size / element_bytes;
    plan.dispatch_x = plan.element_count / 64 + (plan.element_count % 64 != 0);
    return plan;
}

} // namespace VideoCore
