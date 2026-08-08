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

struct PresentFrameTransitions {
    PresentFrameTransition before{};
    PresentFrameTransition after{};
    bool capture{};
};

constexpr PresentFrameTransition GetPresentFrameTransition(const bool is_reusing_frame) {
    if (is_reusing_frame) {
        return {
            .required = false,
            .src_stage = vk::PipelineStageFlagBits2::eFragmentShader,
            .src_access = vk::AccessFlagBits2::eShaderRead,
            .dst_stage = vk::PipelineStageFlagBits2::eFragmentShader,
            .dst_access = vk::AccessFlagBits2::eShaderRead,
            .old_layout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .new_layout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };
    }

    return {
        .required = true,
        .src_stage = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .src_access = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dst_stage = vk::PipelineStageFlagBits2::eFragmentShader,
        .dst_access = vk::AccessFlagBits2::eShaderRead,
        .old_layout = vk::ImageLayout::eGeneral,
        .new_layout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
}

constexpr PresentFrameTransitions GetPresentFrameTransitions(const bool is_reusing_frame,
                                                             const bool capture_requested) {
    if (is_reusing_frame || !capture_requested) {
        return {
            .before = GetPresentFrameTransition(is_reusing_frame),
        };
    }
    return {
        .before =
            {
                .required = true,
                .src_stage = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .src_access = vk::AccessFlagBits2::eColorAttachmentWrite,
                .dst_stage = vk::PipelineStageFlagBits2::eTransfer,
                .dst_access = vk::AccessFlagBits2::eTransferRead,
                .old_layout = vk::ImageLayout::eGeneral,
                .new_layout = vk::ImageLayout::eTransferSrcOptimal,
            },
        .after =
            {
                .required = true,
                .src_stage = vk::PipelineStageFlagBits2::eTransfer,
                .src_access = vk::AccessFlagBits2::eTransferRead,
                .dst_stage = vk::PipelineStageFlagBits2::eFragmentShader,
                .dst_access = vk::AccessFlagBits2::eShaderRead,
                .old_layout = vk::ImageLayout::eTransferSrcOptimal,
                .new_layout = vk::ImageLayout::eShaderReadOnlyOptimal,
            },
        .capture = true,
    };
}

constexpr PresentFrameTransition GetPpInputShadowCaptureTransition(const bool capture_requested) {
    if (!capture_requested) {
        return {.required = false};
    }
    return {
        .required = true,
        .src_stage = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .src_access = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dst_stage = vk::PipelineStageFlagBits2::eTransfer,
        .dst_access = vk::AccessFlagBits2::eTransferRead,
        .old_layout = vk::ImageLayout::eColorAttachmentOptimal,
        .new_layout = vk::ImageLayout::eTransferSrcOptimal,
    };
}

} // namespace Vulkan
