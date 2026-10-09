// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace {

std::string ReadText(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

TEST(AttachmentFeedbackLoop, DepthImagesReceiveFeedbackLoopUsage) {
    const auto source = ReadText(SHADPS4_IMAGE_SOURCE_PATH);
    const auto usage_begin = source.find("static vk::ImageUsageFlags ImageUsageFlags(");
    const auto depth_branch = source.find("if (info.props.is_depth)", usage_begin);
    const auto feedback_usage = source.find("eAttachmentFeedbackLoopEXT", usage_begin);
    ASSERT_NE(usage_begin, std::string::npos);
    ASSERT_NE(depth_branch, std::string::npos);
    ASSERT_NE(feedback_usage, std::string::npos);
    EXPECT_LT(feedback_usage, depth_branch);
}

TEST(AttachmentFeedbackLoop, DepthTargetsUseFeedbackLayoutWhenSampled) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto depth_begin = source.find("if (auto image_id = db_desc.first; image_id)");
    const auto depth_end = source.find("state.depth_stencil_attachment = {};", depth_begin);
    ASSERT_NE(depth_begin, std::string::npos);
    ASSERT_NE(depth_end, std::string::npos);
    const auto depth_path = source.substr(depth_begin, depth_end - depth_begin);

    EXPECT_NE(depth_path.find("if (image.binding.is_bound)"), std::string::npos);
    EXPECT_NE(depth_path.find("eAttachmentFeedbackLoopOptimalEXT"), std::string::npos);
    EXPECT_NE(depth_path.find("attachment_feedback_loop |= image.aspect_mask"),
              std::string::npos);
}

TEST(AttachmentFeedbackLoop, DynamicStatePreservesRequiredAspects) {
    const auto header = ReadText(SHADPS4_SCHEDULER_HEADER_PATH);
    const auto source = ReadText(SHADPS4_SCHEDULER_SOURCE_PATH);

    EXPECT_NE(header.find("vk::ImageAspectFlags feedback_loop_aspects"), std::string::npos);
    EXPECT_NE(header.find(
                  "SetAttachmentFeedbackLoopAspects(const vk::ImageAspectFlags aspects)"),
              std::string::npos);
    EXPECT_NE(source.find("setAttachmentFeedbackLoopEnableEXT(feedback_loop_aspects)"),
              std::string::npos);
}

} // namespace
