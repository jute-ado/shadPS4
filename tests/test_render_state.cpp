// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <new>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/image.h"

TEST(RenderState, DefaultConstructionClearsAttachmentState) {
    alignas(Vulkan::RenderState)
        std::array<std::byte, sizeof(Vulkan::RenderState)> storage;
    std::ranges::fill(storage, std::byte{0xFF});

    auto* state = ::new (storage.data()) Vulkan::RenderState;

    EXPECT_EQ(state->width, 0);
    EXPECT_EQ(state->height, 0);
    EXPECT_EQ(state->num_layers, 0);
    EXPECT_EQ(state->num_color_attachments, 0);
    EXPECT_FALSE(state->depth_stencil_attachment.has_depth);
    EXPECT_FALSE(state->depth_stencil_attachment.depth_clear);
    EXPECT_FALSE(state->depth_stencil_attachment.has_stencil);
    EXPECT_FALSE(state->depth_stencil_attachment.stencil_clear);
    for (const auto& attachment : state->color_attachments) {
        EXPECT_FALSE(attachment.is_clear);
        EXPECT_EQ(attachment.image_view, vk::ImageView{});
        EXPECT_EQ(attachment.image_layout, vk::ImageLayout{});
        EXPECT_EQ(attachment.clear_value, (std::array<u32, 4>{}));
    }

    std::destroy_at(state);
}

TEST(ImageBarrier, RecognizesAttachmentWritesAsHazards) {
    EXPECT_TRUE(VideoCore::IsImageWriteAccess(vk::AccessFlagBits2::eColorAttachmentWrite));
    EXPECT_TRUE(VideoCore::IsImageWriteAccess(vk::AccessFlagBits2::eDepthStencilAttachmentWrite));
}

TEST(ImageBarrier, DoesNotTreatReadOnlyAccessAsAWriteHazard) {
    EXPECT_FALSE(VideoCore::IsImageWriteAccess(vk::AccessFlagBits2::eShaderRead));
    EXPECT_FALSE(VideoCore::IsImageWriteAccess(vk::AccessFlagBits2::eTransferRead));
}
