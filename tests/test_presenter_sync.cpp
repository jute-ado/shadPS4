// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/presenter_sync.h"
#include "video_core/renderer_vulkan/vk_submit_info.h"

TEST(PresenterSync, MakesCompletedFrameWritesVisibleToFragmentSampling) {
    const auto barrier = Vulkan::FrameToPresentationBarrier({});

    EXPECT_EQ(barrier.srcStageMask, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    EXPECT_EQ(barrier.srcAccessMask, vk::AccessFlagBits2::eColorAttachmentWrite);
    EXPECT_EQ(barrier.dstStageMask, vk::PipelineStageFlagBits2::eFragmentShader);
    EXPECT_EQ(barrier.dstAccessMask, vk::AccessFlagBits2::eShaderRead);
    EXPECT_EQ(barrier.oldLayout, vk::ImageLayout::eGeneral);
    EXPECT_EQ(barrier.newLayout, vk::ImageLayout::eShaderReadOnlyOptimal);
}

TEST(PresenterSync, WaitsForCompletedFrameBeforeFragmentSampling) {
    Vulkan::SubmitInfo submit{};

    submit.AddWait({}, 42, vk::PipelineStageFlagBits::eFragmentShader);

    ASSERT_EQ(submit.num_wait_semas, 1u);
    EXPECT_EQ(submit.wait_ticks[0], 42u);
    EXPECT_EQ(submit.wait_stage_masks[0], vk::PipelineStageFlagBits::eFragmentShader);
}
