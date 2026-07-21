// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/descriptor_type.h"

TEST(DescriptorType, ReadOnlyShaderImageUsesSampledDescriptor) {
    EXPECT_EQ(Vulkan::ImageDescriptorType(false), vk::DescriptorType::eSampledImage);
}

TEST(DescriptorType, WritableShaderImageUsesStorageDescriptorEvenWhenUnbound) {
    EXPECT_EQ(Vulkan::ImageDescriptorType(true), vk::DescriptorType::eStorageImage);
}
