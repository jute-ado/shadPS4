// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_scheduler.h"

TEST(SubmitWaitStage, KeepsEachWaitsStageWithItsSemaphore) {
    Vulkan::SubmitInfo info{};
    const vk::Semaphore acquired{reinterpret_cast<VkSemaphore>(1)};
    const vk::Semaphore frame_ready{reinterpret_cast<VkSemaphore>(2)};

    info.AddWait(acquired, 1, vk::PipelineStageFlagBits::eColorAttachmentOutput);
    info.AddWait(frame_ready, 7, vk::PipelineStageFlagBits::eAllCommands);

    ASSERT_EQ(info.num_wait_semas, 2);
    EXPECT_EQ(info.wait_semas[0], acquired);
    EXPECT_EQ(info.wait_ticks[0], 1);
    EXPECT_EQ(info.wait_stage_masks[0], vk::PipelineStageFlagBits::eColorAttachmentOutput);
    EXPECT_EQ(info.wait_semas[1], frame_ready);
    EXPECT_EQ(info.wait_ticks[1], 7);
    EXPECT_EQ(info.wait_stage_masks[1], vk::PipelineStageFlagBits::eAllCommands);
}
