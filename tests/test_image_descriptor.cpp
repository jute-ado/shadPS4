// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/image_descriptor.h"

TEST(ImageDescriptor, ShaderWriteAccessSelectsStorageForNullImage) {
    EXPECT_EQ(Vulkan::ImageDescriptorKindForShaderAccess(true),
              Vulkan::ImageDescriptorKind::Storage);
}

TEST(ImageDescriptor, ShaderReadAccessSelectsSampledForNullImage) {
    EXPECT_EQ(Vulkan::ImageDescriptorKindForShaderAccess(false),
              Vulkan::ImageDescriptorKind::Sampled);
}
