// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <string>

#include "video_core/renderer_vulkan/pipeline_bind_tracker.h"

TEST(PipelineBindTracker, SuppressesOnlyRedundantBindsWithinOneCommandBuffer) {
    Vulkan::PipelineBindTracker<unsigned> tracker;

    EXPECT_TRUE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Graphics, 11));
    EXPECT_FALSE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Graphics, 11));
    EXPECT_TRUE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Graphics, 12));
    EXPECT_FALSE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Graphics, 12));

    EXPECT_TRUE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Compute, 12));
    EXPECT_FALSE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Compute, 12));

    tracker.Reset();

    EXPECT_TRUE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Graphics, 12));
    EXPECT_TRUE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Compute, 12));
}

TEST(GraphicsPipelineDynamicState,
     DeclaresAttachmentFeedbackLoopBeforeIssuingItsDynamicCommand) {
    std::ifstream stream{VK_GRAPHICS_PIPELINE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const std::string source{std::istreambuf_iterator<char>{stream}, {}};

    const auto support_gate =
        source.find("if (instance.IsAttachmentFeedbackLoopLayoutSupported())");
    ASSERT_NE(support_gate, std::string::npos);
    const auto declaration = source.find(
        "dynamic_states.push_back(vk::DynamicState::eAttachmentFeedbackLoopEnableEXT);",
        support_gate);
    ASSERT_NE(declaration, std::string::npos);
    const auto dynamic_info =
        source.find("vk::PipelineDynamicStateCreateInfo dynamic_info", support_gate);
    ASSERT_NE(dynamic_info, std::string::npos);
    EXPECT_LT(declaration, dynamic_info);
}
