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

} // namespace
} // namespace Vulkan
