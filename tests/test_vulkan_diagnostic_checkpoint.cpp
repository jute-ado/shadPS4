// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_diagnostic_checkpoint.h"

TEST(VulkanDiagnosticCheckpoint, EnablesOnlyWhenRequestedAndExtensionIsAvailable) {
    EXPECT_TRUE(Vulkan::ShouldEnableDiagnosticCheckpoints(true, true));
    EXPECT_FALSE(Vulkan::ShouldEnableDiagnosticCheckpoints(false, true));
    EXPECT_FALSE(Vulkan::ShouldEnableDiagnosticCheckpoints(true, false));
}

TEST(VulkanDiagnosticCheckpoint, NativeBackendSuppressesCrashLayer) {
    EXPECT_TRUE(Vulkan::ShouldEnableCrashDiagnosticLayer(true, false));
    EXPECT_FALSE(Vulkan::ShouldEnableCrashDiagnosticLayer(true, true));
    EXPECT_FALSE(Vulkan::ShouldEnableCrashDiagnosticLayer(false, false));
}

TEST(VulkanDiagnosticCheckpoint, PipelineHashRoundTripsThroughOpaqueMarker) {
    constexpr u64 pipeline_hash = 0xfedcba9876543210ull;

    const void* marker = Vulkan::EncodeDiagnosticCheckpoint(pipeline_hash);

    EXPECT_EQ(Vulkan::DecodeDiagnosticCheckpoint(marker), pipeline_hash);
}

TEST(VulkanDiagnosticCheckpoint, PipelineBindIdentityIsOptIn) {
    EXPECT_FALSE(Vulkan::ShouldRecordPipelineBindDiagnostics(false, false));
    EXPECT_TRUE(Vulkan::ShouldRecordPipelineBindDiagnostics(true, false));
    EXPECT_TRUE(Vulkan::ShouldRecordPipelineBindDiagnostics(false, true));
}

TEST(VulkanDiagnosticCheckpoint, CommandIdRoundTripsThroughOpaqueMarker) {
    constexpr u64 command_id = 0x123456789abcdef0ull;

    const void* marker = Vulkan::EncodeDiagnosticCommandId(command_id);

    EXPECT_EQ(Vulkan::DecodeDiagnosticCommandId(marker), command_id);
}

TEST(VulkanDiagnosticCheckpoint, ZeroCommandIdRemainsAnUnresolvedMarker) {
    const void* marker = Vulkan::EncodeDiagnosticCommandId(0);

    EXPECT_EQ(Vulkan::DecodeDiagnosticCommandId(marker), 0u);
}

TEST(VulkanDiagnosticCheckpoint, ReportsCheckpointsForEveryDeviceLossSource) {
    EXPECT_TRUE(Vulkan::ShouldReportDiagnosticCheckpoints(
        Vulkan::DeviceLossSource::Presentation, true));
    EXPECT_TRUE(Vulkan::ShouldReportDiagnosticCheckpoints(Vulkan::DeviceLossSource::Submission,
                                                         true));
}

TEST(VulkanDiagnosticCheckpoint, DoesNotQueryUnsupportedCheckpoints) {
    EXPECT_FALSE(Vulkan::ShouldReportDiagnosticCheckpoints(
        Vulkan::DeviceLossSource::Presentation, false));
    EXPECT_FALSE(Vulkan::ShouldReportDiagnosticCheckpoints(Vulkan::DeviceLossSource::Submission,
                                                          false));
}
