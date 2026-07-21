// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <array>

#include "video_core/texture_cache/image_copy_policy.h"

TEST(ImageCopyPolicy, RejectsDifferentDepthStencilFormats) {
    EXPECT_FALSE(VideoCore::CanUseMaintenance8ImageCopy(vk::Format::eD32Sfloat,
                                                        vk::Format::eD32SfloatS8Uint));
    EXPECT_FALSE(
        VideoCore::CanUseMaintenance8ImageCopy(vk::Format::eD16Unorm, vk::Format::eD16UnormS8Uint));
}

TEST(ImageCopyPolicy, AcceptsIdenticalDepthStencilFormats) {
    EXPECT_TRUE(VideoCore::CanUseMaintenance8ImageCopy(vk::Format::eD32SfloatS8Uint,
                                                       vk::Format::eD32SfloatS8Uint));
}

TEST(ImageCopyPolicy, AcceptsMaintenance8DepthColorPairsInEitherDirection) {
    constexpr std::array depth_16_formats{vk::Format::eD16Unorm, vk::Format::eD16UnormS8Uint};
    constexpr std::array color_16_formats{
        vk::Format::eR16Sfloat, vk::Format::eR16Unorm, vk::Format::eR16Snorm,
        vk::Format::eR16Uint,   vk::Format::eR16Sint,
    };
    constexpr std::array depth_24_or_32_formats{
        vk::Format::eX8D24UnormPack32,
        vk::Format::eD24UnormS8Uint,
        vk::Format::eD32Sfloat,
        vk::Format::eD32SfloatS8Uint,
    };
    constexpr std::array color_32_formats{
        vk::Format::eR32Sfloat,
        vk::Format::eR32Uint,
        vk::Format::eR32Sint,
    };

    for (const auto depth : depth_16_formats) {
        for (const auto color : color_16_formats) {
            EXPECT_TRUE(VideoCore::CanUseMaintenance8ImageCopy(depth, color));
            EXPECT_TRUE(VideoCore::CanUseMaintenance8ImageCopy(color, depth));
        }
    }
    for (const auto depth : depth_24_or_32_formats) {
        for (const auto color : color_32_formats) {
            EXPECT_TRUE(VideoCore::CanUseMaintenance8ImageCopy(depth, color));
            EXPECT_TRUE(VideoCore::CanUseMaintenance8ImageCopy(color, depth));
        }
    }
}

TEST(ImageCopyPolicy, RejectsMismatchedDepthColorSizes) {
    EXPECT_FALSE(
        VideoCore::CanUseMaintenance8ImageCopy(vk::Format::eD32Sfloat, vk::Format::eR16Unorm));
    EXPECT_FALSE(
        VideoCore::CanUseMaintenance8ImageCopy(vk::Format::eD16Unorm, vk::Format::eR32Sfloat));
}

TEST(ImageCopyPolicy, RejectsDifferentColorFormats) {
    EXPECT_FALSE(
        VideoCore::CanUseMaintenance8ImageCopy(vk::Format::eR32Sfloat, vk::Format::eR32Uint));
}
