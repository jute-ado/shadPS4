// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/resource.h"
#include "video_core/texture_cache/image_info.h"

namespace VideoCore {
namespace {

constexpr VAddr ParentAddress = 0x100000000ULL;
constexpr u32 ParentBaseSliceSize = 0x200000;
constexpr u32 ParentSliceSize = 0x80000;
constexpr u32 LogicalViewSize = 0x60000;

constexpr u32 ParentMipOffset(const u32 layers) {
    return ParentBaseSliceSize * layers;
}

ImageInfo MakeParent(const u32 layers = 1) {
    ImageInfo parent{};
    parent.pixel_format = vk::Format::eR16G16B16A16Sfloat;
    parent.type = AmdGpu::ImageType::Color2D;
    parent.resources = {.levels = 2, .layers = layers};
    parent.size = {.width = 480, .height = 270, .depth = 1};
    parent.num_bits = 64;
    parent.num_samples = 1;
    parent.pitch = 512;
    parent.tile_mode = AmdGpu::TileMode::Thin2DThin;
    parent.array_mode = AmdGpu::ArrayMode::Array2DTiledThin1;
    parent.guest_address = ParentAddress;
    parent.mips_layout[0] = {
        .size = ParentBaseSliceSize * layers,
        .pitch = 512,
        .height = 512,
        .offset = 0,
    };
    parent.mips_layout[1] = {
        .size = ParentSliceSize * layers,
        .pitch = 256,
        .height = 256,
        .offset = ParentMipOffset(layers),
    };
    parent.guest_size = ParentMipOffset(layers) + parent.mips_layout[1].size;
    return parent;
}

ImageInfo MakeLogicalMipView(const u32 parent_layers = 1, const u32 parent_layer = 0) {
    ImageInfo view{};
    view.pixel_format = vk::Format::eR16G16B16A16Sfloat;
    view.type = AmdGpu::ImageType::Color2D;
    view.resources = {.levels = 1, .layers = 1};
    view.size = {.width = 240, .height = 135, .depth = 1};
    view.num_bits = 64;
    view.num_samples = 1;
    view.pitch = 256;
    view.tile_mode = AmdGpu::TileMode::Thin2DThin;
    view.array_mode = AmdGpu::ArrayMode::Array2DTiledThin1;
    view.guest_address =
        ParentAddress + ParentMipOffset(parent_layers) + ParentSliceSize * parent_layer;
    view.guest_size = LogicalViewSize;
    view.mips_layout[0] = {
        .size = LogicalViewSize,
        .pitch = 256,
        .height = 135,
        .offset = 0,
    };
    return view;
}

TEST(ImageSubresource, AcceptsLogicalMipViewWithinPaddedParentSlice) {
    const auto parent = MakeParent();
    const auto view = MakeLogicalMipView();

    ASSERT_EQ(view.MipOf(parent), 1);
    EXPECT_EQ(view.SliceOf(parent, 1), 0);
}

TEST(ImageSubresource, UsesParentSliceStrideForBaseLayer) {
    const auto parent = MakeParent(2);
    const auto view = MakeLogicalMipView(2, 1);

    ASSERT_EQ(view.MipOf(parent), 1);
    EXPECT_EQ(view.SliceOf(parent, 1), 1);
}

TEST(ImageSubresource, RejectsViewThatCrossesParentSlice) {
    const auto parent = MakeParent(2);
    auto view = MakeLogicalMipView(2, 1);
    view.guest_size = ParentSliceSize + 1;

    ASSERT_EQ(view.MipOf(parent), 1);
    EXPECT_EQ(view.SliceOf(parent, 1), -1);
}

TEST(ImageSubresource, RejectsAddressInsideButNotAtParentSliceBoundary) {
    const auto parent = MakeParent(2);
    auto view = MakeLogicalMipView(2);
    view.guest_address += 0x1000;

    EXPECT_EQ(view.SliceOf(parent, 1), -1);
}

TEST(ImageSubresource, UsesParentSliceStrideForContiguousLayerViews) {
    const auto parent = MakeParent(4);
    auto view = MakeLogicalMipView(4, 2);
    view.resources.layers = 2;
    view.guest_size = ParentSliceSize * view.resources.layers;

    ASSERT_EQ(view.MipOf(parent), 1);
    EXPECT_EQ(view.SliceOf(parent, 1), 2);
}

TEST(ImageSubresource, RejectsShortStrideAcrossMultipleLayers) {
    const auto parent = MakeParent(4);
    auto view = MakeLogicalMipView(4, 2);
    view.resources.layers = 2;
    view.guest_size = LogicalViewSize * view.resources.layers;

    ASSERT_EQ(view.MipOf(parent), 1);
    EXPECT_EQ(view.SliceOf(parent, 1), -1);
}

} // namespace
} // namespace VideoCore
