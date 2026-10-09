// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_command_checkpoint.h"

namespace Vulkan {
namespace {

TEST(CommandCheckpoint, RetainsComputeIdentityAndArguments) {
    const std::array<u64, 6> shaders{0x1234, 0, 0, 0, 0, 0};
    const std::array<u64, 6> arguments{8, 4, 2, 0, 0, 0};
    const auto marker = RecordCommandCheckpoint(CommandCheckpointType::Dispatch, 0x1234,
                                                shaders, arguments);
    const auto checkpoint = FindCommandCheckpoint(marker);
    ASSERT_TRUE(checkpoint.has_value());
    EXPECT_EQ(checkpoint->pipeline_hash, 0x1234U);
    EXPECT_EQ(checkpoint->shader_hashes, shaders);
    EXPECT_EQ(checkpoint->arguments, arguments);
}

TEST(CommandCheckpoint, ComputeCommandsPublishBeginAndCompletionMarkers) {
    std::ifstream input{SHADPS4_VK_RASTERIZER_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    const auto source = contents.str();
    EXPECT_NE(source.find("CommandCheckpointType::Dispatch"), std::string::npos);
    EXPECT_NE(source.find("CommandCheckpointType::DispatchComplete"), std::string::npos);
    EXPECT_NE(source.find("setCheckpointNV"), std::string::npos);
}

} // namespace
} // namespace Vulkan
