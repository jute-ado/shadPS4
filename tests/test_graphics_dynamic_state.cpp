// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_graphics_dynamic_state.h"

TEST(GraphicsDynamicState, DeclaresAttachmentFeedbackLoopWhenSupported) {
    std::vector<vk::DynamicState> states;

    Vulkan::AppendAttachmentFeedbackLoopDynamicState(states, true);

    EXPECT_EQ(std::ranges::count(states, vk::DynamicState::eAttachmentFeedbackLoopEnableEXT), 1);
}

TEST(GraphicsDynamicState, OmitsAttachmentFeedbackLoopWhenUnsupported) {
    std::vector<vk::DynamicState> states;

    Vulkan::AppendAttachmentFeedbackLoopDynamicState(states, false);

    EXPECT_TRUE(states.empty());
}
