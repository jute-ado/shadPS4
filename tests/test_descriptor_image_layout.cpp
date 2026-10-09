// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/descriptor_image_layout.h"

namespace Vulkan {
namespace {

TEST(DescriptorImageLayout, RefreshesEveryBoundDescriptorAfterAttachmentTransitions) {
    std::array image_infos{
        vk::DescriptorImageInfo{VK_NULL_HANDLE, VK_NULL_HANDLE,
                                vk::ImageLayout::eDepthStencilReadOnlyOptimal},
        vk::DescriptorImageInfo{VK_NULL_HANDLE, VK_NULL_HANDLE,
                                vk::ImageLayout::eShaderReadOnlyOptimal},
        vk::DescriptorImageInfo{VK_NULL_HANDLE, VK_NULL_HANDLE,
                                vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    constexpr std::array bindings{
        DescriptorImageLayoutBinding{.descriptor_index = 0, .image_index = 7},
        DescriptorImageLayoutBinding{.descriptor_index = 2, .image_index = 11},
    };

    RefreshDescriptorImageLayouts(image_infos, bindings, [](const u32 image_index) {
        return image_index == 7 ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : vk::ImageLayout::eGeneral;
    });

    EXPECT_EQ(image_infos[0].imageLayout,
              vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal);
    EXPECT_EQ(image_infos[1].imageLayout, vk::ImageLayout::eShaderReadOnlyOptimal);
    EXPECT_EQ(image_infos[2].imageLayout, vk::ImageLayout::eGeneral);
}

TEST(DescriptorImageLayout, SupportsMultipleDescriptorsForTheSameImage) {
    std::array image_infos{
        vk::DescriptorImageInfo{VK_NULL_HANDLE, VK_NULL_HANDLE,
                                vk::ImageLayout::eDepthStencilReadOnlyOptimal},
        vk::DescriptorImageInfo{VK_NULL_HANDLE, VK_NULL_HANDLE,
                                vk::ImageLayout::eDepthStencilReadOnlyOptimal},
    };
    constexpr std::array bindings{
        DescriptorImageLayoutBinding{.descriptor_index = 0, .image_index = 3},
        DescriptorImageLayoutBinding{.descriptor_index = 1, .image_index = 3},
    };

    RefreshDescriptorImageLayouts(image_infos, bindings, [](const u32) {
        return vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal;
    });

    EXPECT_EQ(image_infos[0].imageLayout,
              vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal);
    EXPECT_EQ(image_infos[1].imageLayout,
              vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal);
}

} // namespace
} // namespace Vulkan
