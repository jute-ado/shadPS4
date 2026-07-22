// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#include "core/emulator_settings.h"
#include "video_core/renderer_vulkan/vk_platform.h"

TEST(VulkanLayerSettings, CrashDiagnosticSettingsUseRegisteredLayerName) {
    EXPECT_STREQ(Vulkan::CrashDiagnosticLayerName, "VK_LAYER_LUNARG_crash_diagnostic");
    EXPECT_STREQ(Vulkan::CrashDiagnosticSettingLayerName, Vulkan::CrashDiagnosticLayerName);
    EXPECT_STREQ(Vulkan::CrashDiagnosticProgressSettingName, "instrument_all_commands");
    EXPECT_STREQ(Vulkan::CrashDiagnosticSyncSettingName, "sync_after_commands");
    EXPECT_STREQ(Vulkan::CrashDiagnosticQueueDumpSettingName, "dump_queue_submits");
    EXPECT_STREQ(Vulkan::CrashDiagnosticCommandBufferDumpSettingName,
                 "dump_command_buffers");
    EXPECT_STREQ(Vulkan::CrashDiagnosticCommandDumpSettingName, "dump_commands");
    EXPECT_STREQ(Vulkan::CrashDiagnosticShaderDumpSettingName, "dump_shaders");
    EXPECT_STREQ(Vulkan::CrashDiagnosticPendingDumpMode, "pending");
    EXPECT_STREQ(Vulkan::CrashDiagnosticAllDumpMode, "all");
    EXPECT_STREQ(Vulkan::CrashDiagnosticShaderDumpMode(false), "on_crash");
    EXPECT_STREQ(Vulkan::CrashDiagnosticShaderDumpMode(true), "on_bind");
    EXPECT_FALSE(Vulkan::CrashDiagnosticSynchronizeCommands);
}

TEST(VulkanLayerSettings, ShaderDumpOnBindIsOptInAndSerializable) {
    VulkanSettings settings{};
    EXPECT_FALSE(settings.vkcrash_diagnostic_shader_dump_on_bind.value);

    settings.vkcrash_diagnostic_shader_dump_on_bind.set(true);
    const nlohmann::json serialized = settings;
    const auto restored = serialized.get<VulkanSettings>();

    EXPECT_TRUE(restored.vkcrash_diagnostic_shader_dump_on_bind.value);
}
