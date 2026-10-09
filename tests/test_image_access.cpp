// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/texture_cache/image.h"

TEST(ImageAccess, ClassifiesEveryImageWriteUsedByTheRenderer) {
    EXPECT_TRUE(VideoCore::IsImageWriteAccess(vk::AccessFlagBits2::eTransferWrite));
    EXPECT_TRUE(VideoCore::IsImageWriteAccess(vk::AccessFlagBits2::eShaderWrite));
    EXPECT_TRUE(VideoCore::IsImageWriteAccess(vk::AccessFlagBits2::eColorAttachmentWrite));
    EXPECT_TRUE(VideoCore::IsImageWriteAccess(vk::AccessFlagBits2::eDepthStencilAttachmentWrite));
    EXPECT_TRUE(VideoCore::IsImageWriteAccess(vk::AccessFlagBits2::eMemoryWrite));
}

TEST(ImageAccess, LeavesReadOnlyImageAccessesUnclassified) {
    EXPECT_FALSE(VideoCore::IsImageWriteAccess(vk::AccessFlagBits2::eTransferRead));
    EXPECT_FALSE(VideoCore::IsImageWriteAccess(vk::AccessFlagBits2::eShaderRead));
    EXPECT_FALSE(VideoCore::IsImageWriteAccess(vk::AccessFlagBits2::eColorAttachmentRead));
    EXPECT_FALSE(VideoCore::IsImageWriteAccess(vk::AccessFlagBits2::eDepthStencilAttachmentRead));
    EXPECT_FALSE(VideoCore::IsImageWriteAccess(vk::AccessFlagBits2::eMemoryRead));
}

TEST(ImageAccess, PreservesWriteClassificationForCombinedAccess) {
    constexpr auto ReadWrite =
        vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
    EXPECT_TRUE(VideoCore::IsImageWriteAccess(ReadWrite));
}
