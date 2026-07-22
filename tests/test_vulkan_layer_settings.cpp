// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#include "video_core/renderer_vulkan/vk_platform.h"

TEST(VulkanLayerSettings, CrashDiagnosticSettingsUseRegisteredLayerName) {
    EXPECT_STREQ(Vulkan::CrashDiagnosticLayerName, "VK_LAYER_LUNARG_crash_diagnostic");
    EXPECT_STREQ(Vulkan::CrashDiagnosticSettingLayerName, Vulkan::CrashDiagnosticLayerName);
    EXPECT_STREQ(Vulkan::CrashDiagnosticProgressSettingName, "instrument_all_commands");
    EXPECT_STREQ(Vulkan::CrashDiagnosticShaderDumpSettingName, "dump_shaders");
    EXPECT_STREQ(Vulkan::CrashDiagnosticShaderDumpMode, "on_crash");
}
