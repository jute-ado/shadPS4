// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/device_feature_policy.h"

TEST(DeviceFeaturePolicy, EnablesSupportedIndirectFirstInstance) {
    vk::PhysicalDeviceFeatures supported{};
    supported.drawIndirectFirstInstance = vk::True;

    const auto enabled = Vulkan::BuildCoreDeviceFeatures(supported);

    EXPECT_EQ(enabled.drawIndirectFirstInstance, vk::True);
}

TEST(DeviceFeaturePolicy, DoesNotRequestUnsupportedIndirectFirstInstance) {
    const vk::PhysicalDeviceFeatures supported{};

    const auto enabled = Vulkan::BuildCoreDeviceFeatures(supported);

    EXPECT_EQ(enabled.drawIndirectFirstInstance, vk::False);
}
