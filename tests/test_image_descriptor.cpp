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

TEST(ImageDescriptor, StorageAttachmentAccessIncludesBothWriters) {
    const auto access = Vulkan::ImageAccessForBinding(
        Vulkan::ImageBindingAccess{
            .is_storage = true,
            .is_target = true,
        });

    EXPECT_TRUE(static_cast<bool>(access & vk::AccessFlagBits2::eShaderRead));
    EXPECT_TRUE(static_cast<bool>(access & vk::AccessFlagBits2::eShaderWrite));
    EXPECT_TRUE(static_cast<bool>(access & vk::AccessFlagBits2::eColorAttachmentRead));
    EXPECT_TRUE(static_cast<bool>(access & vk::AccessFlagBits2::eColorAttachmentWrite));
}

TEST(ImageDescriptor, StorageOnlyAccessDoesNotClaimAttachmentWrites) {
    const auto access = Vulkan::ImageAccessForBinding(
        Vulkan::ImageBindingAccess{
            .is_storage = true,
            .is_target = false,
        });

    EXPECT_EQ(access,
              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
}
