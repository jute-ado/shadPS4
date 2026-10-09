// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "video_core/amdgpu/resource.h"
#include "video_core/descriptor_binding_cursor.h"
#include "video_core/texture_cache/types.h"

namespace {

using VideoCore::SubresourceExtent;

std::string ReadText(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

TEST(ImageDescriptorLayers, NonArrayTypesIgnoreUnusedDepthBits) {
    for (const auto type : {AmdGpu::ImageType::Color1D, AmdGpu::ImageType::Color2D,
                            AmdGpu::ImageType::Color2DMsaa}) {
        AmdGpu::Image image{};
        image.type = static_cast<u64>(type);
        image.depth = 3277;
        image.pow2pad = 1;
        EXPECT_EQ(image.NumLayers(), 1U) << AmdGpu::NameOf(type);
    }
}

TEST(ImageDescriptorLayers, ArrayAndCubeTypesRetainEncodedLayerCounts) {
    AmdGpu::Image array{};
    array.type = static_cast<u64>(AmdGpu::ImageType::Color2DArray);
    array.depth = 4;
    EXPECT_EQ(array.NumLayers(), 5U);

    AmdGpu::Image cube{};
    cube.type = static_cast<u64>(AmdGpu::ImageType::Cube);
    cube.depth = 1;
    EXPECT_EQ(cube.NumLayers(), 12U);
}

TEST(ImageDescriptorSwizzle, RejectsReservedSelectorsBeforeVulkanConversion) {
    AmdGpu::Image image = AmdGpu::Image::Null(false);
    EXPECT_TRUE(image.HasValidDstSelect());

    image.dst_sel_z = 2;
    EXPECT_FALSE(image.HasValidDstSelect());

    image.dst_sel_z = 3;
    EXPECT_FALSE(image.HasValidDstSelect());
}

TEST(ImageDescriptorFormat, RejectsUnsupportedPackedNumberFormatBeforeImageCreation) {
    EXPECT_TRUE(AmdGpu::IsSupportedImageFormatPair(AmdGpu::DataFormat::Format10_11_11,
                                                   AmdGpu::NumberFormat::Float));
    EXPECT_FALSE(AmdGpu::IsSupportedImageFormatPair(AmdGpu::DataFormat::Format10_11_11,
                                                    AmdGpu::NumberFormat::Unorm));
}

TEST(SubresourceExtent, ContainsEqualExtent) {
    constexpr SubresourceExtent available{.levels = 3, .layers = 6};

    EXPECT_TRUE(available.CanContain(SubresourceExtent{.levels = 3, .layers = 6}));
}

TEST(SubresourceExtent, ContainsSmallerExtentInBothDimensions) {
    constexpr SubresourceExtent available{.levels = 3, .layers = 6};

    EXPECT_TRUE(available.CanContain(SubresourceExtent{.levels = 2, .layers = 5}));
}

TEST(SubresourceExtent, RejectsMoreMipLevelsDespiteFewerLayers) {
    constexpr SubresourceExtent available{.levels = 1, .layers = 6};

    EXPECT_FALSE(available.CanContain(SubresourceExtent{.levels = 3, .layers = 1}));
}

TEST(SubresourceExtent, RejectsMoreLayersDespiteFewerMipLevels) {
    constexpr SubresourceExtent available{.levels = 3, .layers = 1};

    EXPECT_FALSE(available.CanContain(SubresourceExtent{.levels = 1, .layers = 6}));
}

TEST(TextureBindingBookkeeping, InvalidBindingsPreserveDescriptorArrayShape) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto function_begin = source.find("void Rasterizer::BindTextures(");
    const auto function_end = source.find("RenderState Rasterizer::BeginRendering(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto helper_begin = body.find("const auto append_invalid_image_binding");
    ASSERT_NE(helper_begin, std::string::npos);
    const auto helper_end = body.find("};", helper_begin);
    ASSERT_NE(helper_end, std::string::npos);
    const auto helper = body.substr(helper_begin, helper_end - helper_begin);
    EXPECT_NE(helper.find("const u32 descriptor_count"), std::string::npos);
    EXPECT_NE(helper.find("i < descriptor_count"), std::string::npos);
    EXPECT_NE(helper.find("image_bindings.emplace_back"), std::string::npos);
    EXPECT_NE(helper.find("image_descriptor_array_sizes.push_back(descriptor_count)"),
              std::string::npos);
    EXPECT_NE(helper.find("image_native_extent_requirements.emplace_back(false, false)"),
              std::string::npos);

    size_t call_count = 0;
    size_t call = helper_end;
    while ((call = body.find("append_invalid_image_binding(image_desc.NumBindings(stage));", call)) !=
           std::string::npos) {
        ++call_count;
        ++call;
    }
    EXPECT_EQ(call_count, 4U);
}

TEST(TextureBindingBookkeeping, UnsupportedFormatPairsAreRejectedBeforeImageConstruction) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto function_begin = source.find("void Rasterizer::BindTextures(");
    const auto function_end = source.find("RenderState Rasterizer::BeginRendering(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto format_check = body.find("AmdGpu::IsSupportedImageFormatPair(data_fmt, num_fmt)");
    const auto image_construction = body.find("VideoCore::TextureCache::ImageDesc full_desc");
    ASSERT_NE(format_check, std::string::npos);
    ASSERT_NE(image_construction, std::string::npos);
    EXPECT_LT(format_check, image_construction);
}

} // namespace

TEST(DescriptorBindingCursor, ArraysConsumeOneBindingAndAllDescriptorCapacity) {
    VideoCore::DescriptorBindingCursor cursor;
    EXPECT_EQ(cursor.Add(), 0U);
    EXPECT_EQ(cursor.Add(6), 1U);
    EXPECT_EQ(cursor.Add(), 2U);
    EXPECT_EQ(cursor.BindingCount(), 3U);
    EXPECT_EQ(cursor.DescriptorCount(), 8U);
    EXPECT_TRUE(cursor.CanUsePushDescriptors(8));
    EXPECT_FALSE(cursor.CanUsePushDescriptors(7));
}
