// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

struct PresentFrameTransition {
    bool required;
    vk::PipelineStageFlags2 src_stage;
    vk::AccessFlags2 src_access;
    vk::PipelineStageFlags2 dst_stage;
    vk::AccessFlags2 dst_access;
    vk::ImageLayout old_layout;
    vk::ImageLayout new_layout;
};

constexpr PresentFrameTransition GetPresentFrameTransition(const bool is_reusing_frame) {
    return {
        .required = true,
        .src_stage = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .src_access = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dst_stage = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dst_access = vk::AccessFlagBits2::eColorAttachmentRead,
        .old_layout = vk::ImageLayout::eGeneral,
        .new_layout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
}

} // namespace Vulkan
