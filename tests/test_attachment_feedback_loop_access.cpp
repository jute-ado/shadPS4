// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/attachment_feedback_loop_access.h"

TEST(AttachmentFeedbackLoopAccess, PreservesEverySimultaneousImageUse) {
    const auto access = Vulkan::AttachmentFeedbackLoopAccess();

    EXPECT_TRUE(bool(access & vk::AccessFlagBits2::eShaderRead));
    EXPECT_TRUE(bool(access & vk::AccessFlagBits2::eColorAttachmentRead));
    EXPECT_TRUE(bool(access & vk::AccessFlagBits2::eColorAttachmentWrite));
    EXPECT_EQ(access, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eColorAttachmentRead |
                          vk::AccessFlagBits2::eColorAttachmentWrite);
}

TEST(AttachmentFeedbackLoopAccess, TextureAndAttachmentPathsUseTheSameContract) {
    EXPECT_EQ(Vulkan::AttachmentFeedbackLoopTextureAccess(),
              Vulkan::AttachmentFeedbackLoopAccess());
    EXPECT_EQ(Vulkan::AttachmentFeedbackLoopRenderTargetAccess(),
              Vulkan::AttachmentFeedbackLoopAccess());
}
