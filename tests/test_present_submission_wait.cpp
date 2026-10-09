// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/present_submission_wait.h"
#include "video_core/renderer_vulkan/vk_submit_info.h"

namespace Vulkan {
namespace {

TEST(PresentSubmissionWait, WaitsForSwapchainAvailabilityAtItsFirstColorWrite) {
    const auto stages = GetPresentSubmissionWaitStages();

    EXPECT_EQ(stages.swapchain_acquire, vk::PipelineStageFlagBits::eColorAttachmentOutput);
}

TEST(PresentSubmissionWait, WaitsForFrameReadinessAtItsFirstFragmentRead) {
    const auto stages = GetPresentSubmissionWaitStages();

    EXPECT_EQ(stages.frame_ready, vk::PipelineStageFlagBits::eFragmentShader);
}

TEST(PresentSubmissionWait, RetainsOneStageMaskForEveryWaitInCallOrder) {
    SubmitInfo submit{};

    submit.AddWait({}, 11, vk::PipelineStageFlagBits::eTransfer);
    submit.AddWait({}, 22, vk::PipelineStageFlagBits::eFragmentShader);
    submit.AddWait({}, 33, vk::PipelineStageFlagBits::eAllCommands);

    ASSERT_EQ(submit.num_wait_semas, 3u);
    EXPECT_EQ(submit.wait_ticks[0], 11u);
    EXPECT_EQ(submit.wait_ticks[1], 22u);
    EXPECT_EQ(submit.wait_ticks[2], 33u);
    EXPECT_EQ(submit.wait_stage_masks[0], vk::PipelineStageFlagBits::eTransfer);
    EXPECT_EQ(submit.wait_stage_masks[1], vk::PipelineStageFlagBits::eFragmentShader);
    EXPECT_EQ(submit.wait_stage_masks[2], vk::PipelineStageFlagBits::eAllCommands);
}

TEST(PresentSubmissionWait, DoesNotInferMeaningFromWaitPosition) {
    SubmitInfo submit{};
    const auto stages = GetPresentSubmissionWaitStages();

    submit.AddWait({}, 42, stages.frame_ready);
    submit.AddWait({}, 1, stages.swapchain_acquire);

    ASSERT_EQ(submit.num_wait_semas, 2u);
    EXPECT_EQ(submit.wait_stage_masks[0], vk::PipelineStageFlagBits::eFragmentShader);
    EXPECT_EQ(submit.wait_stage_masks[1], vk::PipelineStageFlagBits::eColorAttachmentOutput);
}

} // namespace
} // namespace Vulkan
