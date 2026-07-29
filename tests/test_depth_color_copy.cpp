// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/texture_cache/depth_color_copy.h"

namespace {

constexpr bool CanCopyWithMaintenance8(vk::Format src_format, bool src_is_depth,
                                       vk::Format dst_format, bool dst_is_depth) {
    return VideoCore::CanCopyImageDirectly(true, src_is_depth, src_format, dst_is_depth,
                                           dst_format);
}

} // namespace

TEST(DepthColorCopy, AllowsSameAspectCopiesWithoutMaintenance8) {
    EXPECT_TRUE(VideoCore::CanCopyImageDirectly(false, false, vk::Format::eR8G8B8A8Unorm, false,
                                                vk::Format::eR32Uint));
    EXPECT_TRUE(VideoCore::CanCopyImageDirectly(false, true, vk::Format::eD32Sfloat, true,
                                                vk::Format::eD32SfloatS8Uint));
}

TEST(DepthColorCopy, RequiresMaintenance8ForCrossAspectCopies) {
    EXPECT_FALSE(VideoCore::CanCopyImageDirectly(false, false, vk::Format::eR32Sfloat, true,
                                                 vk::Format::eD32SfloatS8Uint));
    EXPECT_FALSE(VideoCore::CanCopyImageDirectly(false, true, vk::Format::eD16Unorm, false,
                                                 vk::Format::eR16Unorm));
}

TEST(DepthColorCopy, AllowsOnlyMaintenance8CompatibleDepthColorFormats) {
    for (const vk::Format depth_format :
         {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
          vk::Format::eX8D24UnormPack32, vk::Format::eD24UnormS8Uint}) {
        for (const vk::Format color_format :
             {vk::Format::eR32Sfloat, vk::Format::eR32Sint, vk::Format::eR32Uint}) {
            EXPECT_TRUE(CanCopyWithMaintenance8(color_format, false, depth_format, true));
            EXPECT_TRUE(CanCopyWithMaintenance8(depth_format, true, color_format, false));
        }
    }

    for (const vk::Format depth_format :
         {vk::Format::eD16Unorm, vk::Format::eD16UnormS8Uint}) {
        for (const vk::Format color_format :
             {vk::Format::eR16Sfloat, vk::Format::eR16Unorm, vk::Format::eR16Snorm,
              vk::Format::eR16Uint, vk::Format::eR16Sint}) {
            EXPECT_TRUE(CanCopyWithMaintenance8(color_format, false, depth_format, true));
            EXPECT_TRUE(CanCopyWithMaintenance8(depth_format, true, color_format, false));
        }
    }

    EXPECT_FALSE(CanCopyWithMaintenance8(vk::Format::eR8G8B8A8Unorm, false,
                                         vk::Format::eD32SfloatS8Uint, true));
    EXPECT_FALSE(CanCopyWithMaintenance8(vk::Format::eD32SfloatS8Uint, true,
                                         vk::Format::eR8G8B8A8Unorm, false));
    EXPECT_FALSE(CanCopyWithMaintenance8(vk::Format::eR16G16Sfloat, false,
                                         vk::Format::eD32Sfloat, true));
}
