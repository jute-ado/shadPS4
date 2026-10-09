// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <fstream>
#include <optional>
#include <sstream>
#include <string>

#include "video_core/renderer_vulkan/descriptor_image_layout.h"

namespace Vulkan {
namespace {

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::string ReadDescriptorSource() {
    return ReadSource(SHADPS4_DESCRIPTOR_RASTERIZER_SOURCE_PATH);
}

std::string DescriptorFunctionBody(const std::string& source, std::string_view begin_marker,
                                   std::string_view end_marker) {
    const auto begin = source.find(begin_marker);
    const auto end = source.find(end_marker, begin);
    EXPECT_NE(begin, std::string::npos);
    EXPECT_NE(end, std::string::npos);
    return begin == std::string::npos || end == std::string::npos
               ? std::string{}
               : source.substr(begin, end - begin);
}

TEST(DescriptorImageLayout, FinalizesViewAndLayoutTogetherAfterBackingSwitch) {
    const vk::ImageView old_view{reinterpret_cast<VkImageView>(uintptr_t{17})};
    const vk::ImageView new_view{reinterpret_cast<VkImageView>(uintptr_t{29})};
    std::array image_infos{
        vk::DescriptorImageInfo{VK_NULL_HANDLE, old_view,
                                vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    constexpr std::array bindings{
        DescriptorImageLayoutBinding{.descriptor_index = 0, .image_index = 7},
    };

    const bool finalized = FinalizeDescriptorImageInfos(
        image_infos, bindings, [&](u32 image_index) -> std::optional<ResolvedDescriptorImage> {
            EXPECT_EQ(image_index, 7U);
            return ResolvedDescriptorImage{new_view, vk::ImageLayout::eGeneral};
        });

    ASSERT_TRUE(finalized);
    EXPECT_EQ(image_infos[0].imageView, new_view);
    EXPECT_EQ(image_infos[0].imageLayout, vk::ImageLayout::eGeneral);
}

TEST(DescriptorImageLayout, FinalizesAReplacementCanonicalImageIdentity) {
    const vk::ImageView old_view{reinterpret_cast<VkImageView>(uintptr_t{31})};
    const vk::ImageView replacement_view{reinterpret_cast<VkImageView>(uintptr_t{37})};
    std::array image_infos{
        vk::DescriptorImageInfo{VK_NULL_HANDLE, old_view,
                                vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    constexpr std::array bindings{
        DescriptorImageLayoutBinding{.descriptor_index = 0, .image_index = 19},
    };

    const bool finalized = FinalizeDescriptorImageInfos(
        image_infos, bindings, [&](u32 logical_binding) {
            EXPECT_EQ(logical_binding, 19U);
            return std::optional{
                ResolvedDescriptorImage{replacement_view, vk::ImageLayout::eGeneral}};
        });

    ASSERT_TRUE(finalized);
    EXPECT_EQ(image_infos[0].imageView, replacement_view);
    EXPECT_EQ(image_infos[0].imageLayout, vk::ImageLayout::eGeneral);
}

TEST(DescriptorImageLayout, FailedFinalResolutionPublishesNothing) {
    const vk::ImageView first_old{reinterpret_cast<VkImageView>(uintptr_t{41})};
    const vk::ImageView second_old{reinterpret_cast<VkImageView>(uintptr_t{43})};
    const vk::ImageView first_new{reinterpret_cast<VkImageView>(uintptr_t{47})};
    std::array image_infos{
        vk::DescriptorImageInfo{VK_NULL_HANDLE, first_old,
                                vk::ImageLayout::eShaderReadOnlyOptimal},
        vk::DescriptorImageInfo{VK_NULL_HANDLE, second_old,
                                vk::ImageLayout::eDepthStencilReadOnlyOptimal},
    };
    constexpr std::array bindings{
        DescriptorImageLayoutBinding{.descriptor_index = 0, .image_index = 2},
        DescriptorImageLayoutBinding{.descriptor_index = 1, .image_index = 3},
    };

    const bool finalized = FinalizeDescriptorImageInfos(
        image_infos, bindings,
        [&](u32 logical_binding) -> std::optional<ResolvedDescriptorImage> {
            if (logical_binding == 2) {
                return ResolvedDescriptorImage{first_new, vk::ImageLayout::eGeneral};
            }
            return std::nullopt;
        });

    EXPECT_FALSE(finalized);
    EXPECT_EQ(image_infos[0].imageView, first_old);
    EXPECT_EQ(image_infos[0].imageLayout, vk::ImageLayout::eShaderReadOnlyOptimal);
    EXPECT_EQ(image_infos[1].imageView, second_old);
    EXPECT_EQ(image_infos[1].imageLayout, vk::ImageLayout::eDepthStencilReadOnlyOptimal);
}

TEST(DescriptorImageLayout, ProductionFinalizesAfterEveryWaitCapableDrawPreparation) {
    const std::string source = ReadDescriptorSource();
    const std::string bind = DescriptorFunctionBody(
        source, "bool Rasterizer::BindResources(", "void Rasterizer::PrepareAddr64PointerResidency(");
    const std::string textures = DescriptorFunctionBody(
        source, "void Rasterizer::BindTextures(", "bool Rasterizer::FinalizeImageDescriptors(");
    const std::string draw =
        DescriptorFunctionBody(source, "void Rasterizer::Draw(bool", "void Rasterizer::DrawIndirect(");
    const std::string indirect = DescriptorFunctionBody(
        source, "void Rasterizer::DrawIndirect(", "void Rasterizer::DispatchDirect(");

    EXPECT_NE(bind.find("pending_image_bindings.clear()"), std::string::npos);
    EXPECT_EQ(textures.find("pending_image_bindings.clear()"), std::string::npos);
    EXPECT_NE(textures.find("image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE"),
              std::string::npos);
    EXPECT_EQ(textures.find("image_infos.emplace_back(VK_NULL_HANDLE, *image_view"),
              std::string::npos);

    for (const auto* body : {&draw, &indirect}) {
        const auto begin_rendering = body->find("BeginRendering(pipeline, state)");
        const auto late_prepare = body->find("PrepareDrawBuffersThenBindCommandState(");
        const auto final_images = body->find("FinalizeImageDescriptors(", late_prepare);
        const auto commit = body->find("CommitBufferBindings(binding_batch", final_images);
        const auto publish = body->find("pipeline->BindResources(", commit);
        ASSERT_NE(begin_rendering, std::string::npos);
        ASSERT_NE(late_prepare, std::string::npos);
        ASSERT_NE(final_images, std::string::npos);
        ASSERT_NE(commit, std::string::npos);
        ASSERT_NE(publish, std::string::npos);
        // Buffer preparation may wait for the stream buffer and retire/rebind image backings.
        // Capture the attachment handles only after that late mutation window, while retaining
        // the attachment transition before image descriptor publication so feedback-loop layouts
        // are still resolved into the descriptors.
        EXPECT_LT(late_prepare, begin_rendering);
        EXPECT_LT(begin_rendering, final_images);
        EXPECT_LT(final_images, commit);
        EXPECT_LT(commit, publish);
    }
}

TEST(DescriptorImageLayout, ProductionFinalizesComputeImagesAfterLateBufferPreparation) {
    const std::string source = ReadDescriptorSource();
    const std::string direct = DescriptorFunctionBody(
        source, "void Rasterizer::DispatchDirect()", "void Rasterizer::DispatchIndirect(");
    const std::string indirect = DescriptorFunctionBody(
        source, "void Rasterizer::DispatchIndirect(", "u64 Rasterizer::Flush()");

    for (const auto* body : {&direct, &indirect}) {
        const auto bind = body->find("BindResources(pipeline)");
        const auto final_buffers = body->find("FinalizeBufferBindings(binding_batch)", bind);
        const auto final_images = body->find("FinalizeImageDescriptors(", final_buffers);
        const auto commit = body->find("CommitBufferBindings(binding_batch", final_images);
        const auto publish = body->find("pipeline->BindResources(", commit);
        ASSERT_NE(bind, std::string::npos);
        ASSERT_NE(final_buffers, std::string::npos);
        ASSERT_NE(final_images, std::string::npos);
        ASSERT_NE(commit, std::string::npos);
        ASSERT_NE(publish, std::string::npos);
        EXPECT_LT(bind, final_buffers);
        EXPECT_LT(final_buffers, final_images);
        EXPECT_LT(final_images, commit);
        EXPECT_LT(commit, publish);
    }
}

TEST(DescriptorImageLayout, FinalProductionResolutionCannotReplaceCanonicalImagesOrWait) {
    const std::string source = ReadDescriptorSource();
    const std::string body = DescriptorFunctionBody(
        source, "bool Rasterizer::FinalizeImageDescriptors(", "void Rasterizer::TrackBoundImage(");

    EXPECT_NE(body.find("TryGetImage("), std::string::npos);
    EXPECT_NE(body.find("image_uid"), std::string::npos);
    EXPECT_NE(body.find("FindView(pending.view_info, false)"), std::string::npos);
    for (std::string_view forbidden : {"FindImage(", "FindTexture(", "UpdateImage(",
                                       "SetBackingSamples(", "SetHostScale(", "scheduler."}) {
        EXPECT_EQ(body.find(forbidden), std::string::npos) << forbidden;
    }
}

TEST(DescriptorImageLayout, MissingViewAbortsBeforeOptionalViewDereference) {
    const std::string source = ReadDescriptorSource();
    const std::string body = DescriptorFunctionBody(
        source, "bool Rasterizer::FinalizeImageDescriptors(", "void Rasterizer::TrackBoundImage(");

    const auto missing_view = body.find("if (!image_view.image_view)");
    ASSERT_NE(missing_view, std::string::npos);
    const auto missing_view_return = body.find("return std::nullopt;", missing_view);
    const auto optional_dereference = body.find("*image_view.image_view", missing_view);
    ASSERT_NE(optional_dereference, std::string::npos);
    EXPECT_NE(missing_view_return, std::string::npos);
    EXPECT_LT(missing_view_return, optional_dereference);
}

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
