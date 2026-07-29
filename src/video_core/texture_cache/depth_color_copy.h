// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {

constexpr bool IsMaintenance8DepthColorCopyCompatible(vk::Format depth_format,
                                                       vk::Format color_format) {
    switch (depth_format) {
    case vk::Format::eD32Sfloat:
    case vk::Format::eD32SfloatS8Uint:
    case vk::Format::eX8D24UnormPack32:
    case vk::Format::eD24UnormS8Uint:
        return color_format == vk::Format::eR32Sfloat ||
               color_format == vk::Format::eR32Sint ||
               color_format == vk::Format::eR32Uint;
    case vk::Format::eD16Unorm:
    case vk::Format::eD16UnormS8Uint:
        return color_format == vk::Format::eR16Sfloat ||
               color_format == vk::Format::eR16Unorm ||
               color_format == vk::Format::eR16Snorm ||
               color_format == vk::Format::eR16Uint ||
               color_format == vk::Format::eR16Sint;
    default:
        return false;
    }
}

constexpr bool CanCopyImageDirectly(bool maintenance8_supported, bool src_is_depth,
                                    vk::Format src_format, bool dst_is_depth,
                                    vk::Format dst_format) {
    if (src_is_depth == dst_is_depth) {
        return true;
    }
    if (!maintenance8_supported) {
        return false;
    }

    const vk::Format depth_format = src_is_depth ? src_format : dst_format;
    const vk::Format color_format = src_is_depth ? dst_format : src_format;
    return IsMaintenance8DepthColorCopyCompatible(depth_format, color_format);
}

} // namespace VideoCore
