// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_submit_info.h"

TEST(PresenterWaitStage, PreservesTheConsumerStageForEachSemaphore) {
    Vulkan::SubmitInfo submit{};

    submit.AddWait({}, 42, vk::PipelineStageFlagBits::eFragmentShader);

    ASSERT_EQ(submit.num_wait_semas, 1u);
    EXPECT_EQ(submit.wait_ticks[0], 42u);
    EXPECT_EQ(submit.wait_stage_masks[0], vk::PipelineStageFlagBits::eFragmentShader);
}
