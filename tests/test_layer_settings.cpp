// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_layer_settings.h"

TEST(VulkanLayerSettings, TargetsCrashDiagnosticSettingsAtEnabledLayer) {
    EXPECT_EQ(Vulkan::LayerSettings::CrashDiagnosticLayer,
              "VK_LAYER_LUNARG_crash_diagnostic");
}

TEST(VulkanLayerSettings, CapturesTheFaultingCommandAndShaders) {
    EXPECT_TRUE(Vulkan::LayerSettings::InstrumentAllCrashCommands);
    EXPECT_EQ(Vulkan::LayerSettings::CrashShaderDumpMode, "on_crash");
    EXPECT_EQ(Vulkan::LayerSettings::CrashCommandDumpMode, "all");
    EXPECT_EQ(Vulkan::LayerSettings::CrashCommandBufferDumpMode, "pending");
    EXPECT_EQ(Vulkan::LayerSettings::CrashQueueSubmitDumpMode, "pending");
}
