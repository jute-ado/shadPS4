// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {

namespace ImageCopyPolicy {

constexpr bool IsDepth16(vk::Format format) {
    return format == vk::Format::eD16Unorm || format == vk::Format::eD16UnormS8Uint;
}

constexpr bool IsDepth24Or32(vk::Format format) {
    return format == vk::Format::eX8D24UnormPack32 || format == vk::Format::eD24UnormS8Uint ||
           format == vk::Format::eD32Sfloat || format == vk::Format::eD32SfloatS8Uint;
}

constexpr bool IsColor16(vk::Format format) {
    return format == vk::Format::eR16Sfloat || format == vk::Format::eR16Unorm ||
           format == vk::Format::eR16Snorm || format == vk::Format::eR16Uint ||
           format == vk::Format::eR16Sint;
}

constexpr bool IsColor32(vk::Format format) {
    return format == vk::Format::eR32Sfloat || format == vk::Format::eR32Uint ||
           format == vk::Format::eR32Sint;
}

} // namespace ImageCopyPolicy

// VK_KHR_maintenance8 only relaxes depth-aspect copies when the other image is a
// matching single-component color format. Different depth/stencil formats remain incompatible.
constexpr bool CanUseMaintenance8ImageCopy(vk::Format src, vk::Format dst) {
    if (src == dst) {
        return true;
    }

    using namespace ImageCopyPolicy;
    return (IsDepth16(src) && IsColor16(dst)) || (IsColor16(src) && IsDepth16(dst)) ||
           (IsDepth24Or32(src) && IsColor32(dst)) || (IsColor32(src) && IsDepth24Or32(dst));
}

} // namespace VideoCore
