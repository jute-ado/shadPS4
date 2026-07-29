// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/texture_cache/depth_association.h"

TEST(DepthAssociation, RedirectsOnlyStencilCompatibleViews) {
    EXPECT_TRUE(VideoCore::ShouldUseAssociatedDepthForView(vk::Format::eR8Uint));
    EXPECT_TRUE(VideoCore::ShouldUseAssociatedDepthForView(vk::Format::eR8Unorm));

    EXPECT_FALSE(
        VideoCore::ShouldUseAssociatedDepthForView(vk::Format::eR16G16B16A16Sfloat));
    EXPECT_FALSE(VideoCore::ShouldUseAssociatedDepthForView(vk::Format::eR32Sfloat));
}
