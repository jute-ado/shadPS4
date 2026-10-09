// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

enum class ImageDescriptorKind {
    Sampled,
    Storage,
};

struct ImageBindingAccess {
    bool is_storage;
    bool is_target;
};

[[nodiscard]] constexpr ImageDescriptorKind ImageDescriptorKindForShaderAccess(
    bool shader_writes) {
    return shader_writes ? ImageDescriptorKind::Storage : ImageDescriptorKind::Sampled;
}

[[nodiscard]] constexpr vk::AccessFlags2 ImageAccessForBinding(ImageBindingAccess binding) {
    vk::AccessFlags2 access = vk::AccessFlagBits2::eShaderRead;
    if (binding.is_storage) {
        access |= vk::AccessFlagBits2::eShaderWrite;
    }
    if (binding.is_target) {
        access |= vk::AccessFlagBits2::eColorAttachmentRead |
                  vk::AccessFlagBits2::eColorAttachmentWrite;
    }
    return access;
}

} // namespace Vulkan
