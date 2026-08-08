// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/present_frame_transition.h"

namespace Vulkan {
namespace {

TEST(PresentFrameTransition, MakesNewlyRenderedFrameVisibleToFragmentShaderReads) {
    const auto transition = GetPresentFrameTransition(false);

    EXPECT_TRUE(transition.required);
    EXPECT_EQ(transition.src_stage, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    EXPECT_EQ(transition.src_access, vk::AccessFlagBits2::eColorAttachmentWrite);
    EXPECT_EQ(transition.dst_stage, vk::PipelineStageFlagBits2::eFragmentShader);
    EXPECT_EQ(transition.dst_access, vk::AccessFlagBits2::eShaderRead);
    EXPECT_EQ(transition.old_layout, vk::ImageLayout::eGeneral);
    EXPECT_EQ(transition.new_layout, vk::ImageLayout::eShaderReadOnlyOptimal);
}

TEST(PresentFrameTransition, KeepsReusedFrameInShaderReadOnlyLayout) {
    const auto transition = GetPresentFrameTransition(true);

    EXPECT_FALSE(transition.required);
    EXPECT_EQ(transition.old_layout, vk::ImageLayout::eShaderReadOnlyOptimal);
    EXPECT_EQ(transition.new_layout, vk::ImageLayout::eShaderReadOnlyOptimal);
}

TEST(PresentFrameTransition, CapturesNewPostProcessedFrameBeforeFragmentSampling) {
    const auto transitions = GetPresentFrameTransitions(false, true);

    EXPECT_TRUE(transitions.capture);
    EXPECT_TRUE(transitions.before.required);
    EXPECT_EQ(transitions.before.src_stage, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    EXPECT_EQ(transitions.before.src_access, vk::AccessFlagBits2::eColorAttachmentWrite);
    EXPECT_EQ(transitions.before.dst_stage, vk::PipelineStageFlagBits2::eTransfer);
    EXPECT_EQ(transitions.before.dst_access, vk::AccessFlagBits2::eTransferRead);
    EXPECT_EQ(transitions.before.old_layout, vk::ImageLayout::eGeneral);
    EXPECT_EQ(transitions.before.new_layout, vk::ImageLayout::eTransferSrcOptimal);

    EXPECT_TRUE(transitions.after.required);
    EXPECT_EQ(transitions.after.src_stage, vk::PipelineStageFlagBits2::eTransfer);
    EXPECT_EQ(transitions.after.src_access, vk::AccessFlagBits2::eTransferRead);
    EXPECT_EQ(transitions.after.dst_stage, vk::PipelineStageFlagBits2::eFragmentShader);
    EXPECT_EQ(transitions.after.dst_access, vk::AccessFlagBits2::eShaderRead);
    EXPECT_EQ(transitions.after.old_layout, vk::ImageLayout::eTransferSrcOptimal);
    EXPECT_EQ(transitions.after.new_layout, vk::ImageLayout::eShaderReadOnlyOptimal);
}

TEST(PresentFrameTransition, NeverResamplesAReusedPresentation) {
    const auto transitions = GetPresentFrameTransitions(true, true);
    EXPECT_FALSE(transitions.capture);
    EXPECT_FALSE(transitions.before.required);
    EXPECT_FALSE(transitions.after.required);
    EXPECT_EQ(transitions.before.old_layout, vk::ImageLayout::eShaderReadOnlyOptimal);
    EXPECT_EQ(transitions.before.new_layout, vk::ImageLayout::eShaderReadOnlyOptimal);
}

TEST(PresentFrameTransition, CapturesPpInputShadowAfterItsColorWriteWithoutRestore) {
    const auto transition = GetPpInputShadowCaptureTransition(true);
    EXPECT_TRUE(transition.required);
    EXPECT_EQ(transition.src_stage, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    EXPECT_EQ(transition.src_access, vk::AccessFlagBits2::eColorAttachmentWrite);
    EXPECT_EQ(transition.dst_stage, vk::PipelineStageFlagBits2::eTransfer);
    EXPECT_EQ(transition.dst_access, vk::AccessFlagBits2::eTransferRead);
    EXPECT_EQ(transition.old_layout, vk::ImageLayout::eColorAttachmentOptimal);
    EXPECT_EQ(transition.new_layout, vk::ImageLayout::eTransferSrcOptimal);

    EXPECT_FALSE(GetPpInputShadowCaptureTransition(false).required);
}

} // namespace
} // namespace Vulkan
