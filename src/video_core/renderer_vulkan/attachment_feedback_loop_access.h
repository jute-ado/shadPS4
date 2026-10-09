// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

[[nodiscard]] constexpr vk::AccessFlags2 AttachmentFeedbackLoopAccess() noexcept {
    return vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eColorAttachmentRead |
           vk::AccessFlagBits2::eColorAttachmentWrite;
}

[[nodiscard]] constexpr vk::AccessFlags2 AttachmentFeedbackLoopTextureAccess() noexcept {
    return AttachmentFeedbackLoopAccess();
}

[[nodiscard]] constexpr vk::AccessFlags2 AttachmentFeedbackLoopRenderTargetAccess() noexcept {
    return AttachmentFeedbackLoopAccess();
}

} // namespace Vulkan
