// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include "video_core/amdgpu/resource.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/image_info.h"

namespace VideoCore {

bool ImageInfo::IsCompatible(const ImageInfo& info) const {
    return (IsVulkanFormatCompatible(pixel_format, info.pixel_format) ||
            IsVulkanFormatCompatible(info.pixel_format, pixel_format)) &&
           num_samples == info.num_samples && num_bits == info.num_bits;
}

s32 ImageInfo::MipOf(const ImageInfo& info) const {
    if (!IsCompatible(info)) {
        return -1;
    }

    if (info.array_mode != array_mode) {
        return -1;
    }

    // Currently we expect only one level to be copied.
    if (resources.levels != 1) {
        return -1;
    }

    const auto info_dim = info.props.is_block ? 2 : 0;
    const auto this_dim = props.is_block ? 2 : 0;

    // Find mip
    auto mip = -1;
    for (auto m = 0; m < info.resources.levels; ++m) {
        const auto& [mip_size, mip_pitch, mip_height, mip_ofs] = info.mips_layout[m];
        const VAddr mip_base = info.guest_address + mip_ofs;
        const VAddr mip_end = mip_base + mip_size;
        const u32 slice_size = mip_size / info.resources.layers;
        if (guest_address >= mip_base && guest_address < mip_end &&
            (guest_address - mip_base) % slice_size == 0 &&
            (pitch >> this_dim) == (mip_pitch >> info_dim)) {
            mip = m;
            break;
        }
    }

    if (mip < 0) {
        return -1;
    }

    // 2D block dimensions of both images should be the same.
    const auto mip_w = std::max(info.size.width >> (mip + info_dim), 1u);
    const auto mip_h = std::max(info.size.height >> (mip + info_dim), 1u);
    const auto this_w = std::max(size.width >> this_dim, 1u);
    const auto this_h = std::max(size.height >> this_dim, 1u);
    if ((this_w != mip_w) || (this_h != mip_h)) {
        return -1;
    }

    const auto mip_d = std::max(info.size.depth >> mip, 1u);
    if (info.type == AmdGpu::ImageType::Color3D && type == AmdGpu::ImageType::Color2D) {
        // In case of 2D array to 3D copy, make sure we have proper number of layers.
        if (resources.layers != mip_d) {
            return -1;
        }
    } else {
        if (type != info.type) {
            return -1;
        }
    }

    return mip;
}

s32 ImageInfo::SliceOf(const ImageInfo& info, s32 mip) const {
    if (!IsCompatible(info)) {
        return -1;
    }

    // Array slices should be of the same type.
    if (type != info.type) {
        return -1;
    }

    // 2D block dimensions of both images should be the same.
    const auto info_dim = info.props.is_block ? 2 : 0;
    const auto mip_w = std::max(info.size.width >> (mip + info_dim), 1u);
    const auto mip_h = std::max(info.size.height >> (mip + info_dim), 1u);
    const auto mip_p = std::max(info.mips_layout[mip].pitch >> info_dim, 1u);

    const auto this_dim = props.is_block ? 2 : 0;
    const auto this_w = std::max(size.width >> this_dim, 1u);
    const auto this_h = std::max(size.height >> this_dim, 1u);
    const auto this_p = std::max(pitch >> this_dim, 1u);
    if ((this_w != mip_w) || (this_h != mip_h) || (this_p != mip_p)) {
        return -1;
    }

    // Check for size alignment.
    const u32 slice_size = info.mips_layout[mip].size / info.resources.layers;
    if (guest_size % slice_size != 0) {
        return -1;
    }

    // Ensure that address is aligned too.
    const auto addr_diff = guest_address - (info.guest_address + info.mips_layout[mip].offset);
    if ((addr_diff % guest_size) != 0) {
        return -1;
    }

    return addr_diff / guest_size;
}

} // namespace VideoCore
