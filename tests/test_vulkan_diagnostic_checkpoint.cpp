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
