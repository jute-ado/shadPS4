// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {
namespace {

TEST(SubmitInfo, RetainsExplicitStagePerWaitInInsertionOrder) {
    SubmitInfo info{};
    info.AddWait({}, 1, vk::PipelineStageFlagBits::eColorAttachmentOutput);
    info.AddWait({}, 42,
                 vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eTransfer);

    ASSERT_EQ(info.num_wait_semas, 2u);
    EXPECT_EQ(info.wait_ticks[0], 1u);
    EXPECT_EQ(info.wait_stages[0], vk::PipelineStageFlagBits::eColorAttachmentOutput);
    EXPECT_EQ(info.wait_ticks[1], 42u);
    EXPECT_EQ(info.wait_stages[1],
              vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eTransfer);
}

TEST(SubmitInfo, DefaultWaitStageIsConservativeForExistingCallers) {
    SubmitInfo info{};
    info.AddWait({});
    ASSERT_EQ(info.num_wait_semas, 1u);
    EXPECT_EQ(info.wait_stages[0], vk::PipelineStageFlagBits::eAllCommands);
}

TEST(SubmitInfo, PresenterFrameReadyWaitPreservesNormalStageAndCoversShadowTransfer) {
    EXPECT_EQ(FrameReadyWaitStage(false), vk::PipelineStageFlagBits::eColorAttachmentOutput);
    EXPECT_EQ(FrameReadyWaitStage(true),
              vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eTransfer);
}

} // namespace
} // namespace Vulkan
