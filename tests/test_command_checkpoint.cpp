// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <gtest/gtest.h>
#include "video_core/renderer_vulkan/vk_command_checkpoint.h"

namespace Vulkan {
TEST(CommandCheckpoint, RetainsComputeIdentityAndArguments) {
    const std::array<u64, 6> shaders{0x1234, 0, 0, 0, 0, 0};
    const std::array<u64, 6> arguments{8, 4, 2, 0, 0, 0};
    const auto checkpoint = FindCommandCheckpoint(
        RecordCommandCheckpoint(CommandCheckpointType::Dispatch, 0x1234, shaders, arguments));
    ASSERT_TRUE(checkpoint.has_value());
    EXPECT_EQ(checkpoint->pipeline_hash, 0x1234U);
    EXPECT_EQ(checkpoint->arguments, arguments);
}

TEST(CommandCheckpoint, RetainsBoundResourceDeviceAddresses) {
    const std::array<u64, 6> shaders{};
    const std::array<u64, 6> arguments{};
    const std::array<u64, 8> resources{0x1000, 0x2000, 0x3000, 0, 0, 0, 0, 0};
    const auto checkpoint = FindCommandCheckpoint(RecordCommandCheckpoint(
        CommandCheckpointType::DrawIndirect, 0x1234, shaders, arguments, resources));
    ASSERT_TRUE(checkpoint.has_value());
    EXPECT_EQ(checkpoint->resource_addresses, resources);
}

TEST(CommandCheckpoint, EncodesUpToFourGuestBufferRanges) {
    const std::array<std::pair<u64, u64>, 5> ranges{{
        {0x1000, 0x40},
        {0x2000, 0x80},
        {0x3000, 0x100},
        {0x4000, 0x200},
        {0x5000, 0x400},
    }};

    EXPECT_EQ(EncodeCommandCheckpointBufferRanges(ranges),
              (std::array<u64, 8>{0x1000, 0x40, 0x2000, 0x80, 0x3000, 0x100, 0x4000,
                                  0x200}));
}

TEST(CommandCheckpoint, RetainsFaultParserBeginAndCompletionMarkers) {
    const std::array<u64, 6> arguments{0x2000000, 0x10000000, 0x20000, 0, 0, 0};
    const auto begin = FindCommandCheckpoint(RecordCommandCheckpoint(
        CommandCheckpointType::FaultProcess, 0, {}, arguments));
    const auto complete = FindCommandCheckpoint(RecordCommandCheckpoint(
        CommandCheckpointType::FaultProcessComplete, 0, {}, arguments));

    ASSERT_TRUE(begin.has_value());
    ASSERT_TRUE(complete.has_value());
    EXPECT_EQ(begin->type, CommandCheckpointType::FaultProcess);
    EXPECT_EQ(complete->type, CommandCheckpointType::FaultProcessComplete);
    EXPECT_EQ(complete->arguments, arguments);
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

TEST(CommandCheckpoint, GraphicsCommandsPublishBeginAndCompletionMarkers) {
    std::ifstream input{SHADPS4_VK_RASTERIZER_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    const auto source = contents.str();
    EXPECT_NE(source.find("CommandCheckpointType::Draw"), std::string::npos);
    EXPECT_NE(source.find("CommandCheckpointType::DrawComplete"), std::string::npos);
    EXPECT_NE(source.find("CommandCheckpointType::DrawIndirect"), std::string::npos);
    EXPECT_NE(source.find("CommandCheckpointType::DrawIndirectComplete"), std::string::npos);
}
} // namespace Vulkan
