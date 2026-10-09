// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/amdgpu/resource.h"
#include "shader_recompiler/resource.h"
#include "video_core/descriptor_binding_cursor.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/texture_cache/types.h"

namespace {

using VideoCore::SubresourceExtent;

struct FlatDescriptorInfo {
    std::vector<u32> flattened_ud_buf;

    template <typename T>
    T ReadUdSharp(u32 sharp_idx) const noexcept {
        T value{};
        std::memcpy(&value, flattened_ud_buf.data() + sharp_idx, sizeof(value));
        return value;
    }
};

AmdGpu::Image MakeImage(VAddr address, AmdGpu::ImageType type,
                        AmdGpu::NumberFormat number_format) {
    auto image = AmdGpu::Image::Null(false);
    image.base_address = address >> 8;
    image.type = static_cast<u64>(type);
    image.num_format = static_cast<u64>(number_format);
    return image;
}

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

TEST(TextureBindingBookkeeping, InvalidBindingsAppendAllParallelMetadata) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto function_begin = source.find("void Rasterizer::BindTextures(");
    const auto function_end = source.find("RenderState Rasterizer::BeginRendering(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto helper_begin = body.find("const auto append_invalid_image_slot");
    ASSERT_NE(helper_begin, std::string::npos);
    const auto helper_end = body.find("};", helper_begin);
    ASSERT_NE(helper_end, std::string::npos);
    const auto helper = body.substr(helper_begin, helper_end - helper_begin);
    EXPECT_NE(helper.find("bool is_storage"), std::string::npos);
    EXPECT_NE(helper.find("image_bindings.emplace_back"), std::string::npos);
    EXPECT_NE(helper.find("desc.type = is_storage ?"), std::string::npos);
    EXPECT_NE(helper.find("BindingType::Storage"), std::string::npos);
    EXPECT_NE(helper.find("BindingType::Texture"), std::string::npos);
    EXPECT_NE(helper.find("image_native_extent_requirements.emplace_back(false, false)"),
              std::string::npos);

    size_t call_count = 0;
    size_t call = helper_end;
    while ((call = body.find("append_invalid_image_slot(image_desc.is_written);", call)) !=
           std::string::npos) {
        ++call_count;
        ++call;
    }
    EXPECT_EQ(call_count, 3U);
    EXPECT_NE(body.find("image_descriptor_array_sizes.push_back(num_bindings)"),
              std::string::npos);
    EXPECT_NE(body.find("set_write.descriptorCount = array_size"), std::string::npos);
    EXPECT_NE(body.find("image_binding_idx += array_size"), std::string::npos);
    const auto compatibility_check =
        body.find("image_desc.IsTableEntryCompatible(stage, table_index)");
    const auto read_sharp = body.find("image_desc.GetSharp(stage, table_index)");
    ASSERT_NE(compatibility_check, std::string::npos);
    ASSERT_NE(read_sharp, std::string::npos);
    EXPECT_LT(compatibility_check, read_sharp);
}

TEST(TextureFormatAdmission, DistinguishesSupportedPairsFromEnumValidButUnsupportedPairs) {
    using Vulkan::LiverpoolToVK::IsSurfaceFormatSupported;

    EXPECT_TRUE(IsSurfaceFormatSupported(AmdGpu::DataFormat::Format1_5_5_5,
                                         AmdGpu::NumberFormat::Unorm));
    EXPECT_FALSE(IsSurfaceFormatSupported(AmdGpu::DataFormat::Format1_5_5_5,
                                          AmdGpu::NumberFormat::Snorm));
    EXPECT_FALSE(IsSurfaceFormatSupported(static_cast<AmdGpu::DataFormat>(64),
                                          AmdGpu::NumberFormat::Unorm));
    EXPECT_FALSE(IsSurfaceFormatSupported(AmdGpu::DataFormat::Format8,
                                          static_cast<AmdGpu::NumberFormat>(16)));
}

TEST(TextureFormatAdmission, RejectsUnsupportedPairBeforeConstructingCachedImage) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto function_begin = source.find("void Rasterizer::BindTextures(");
    const auto function_end = source.find("RenderState Rasterizer::BeginRendering(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto compatibility_check =
        body.find("!LiverpoolToVK::IsSurfaceFormatSupported(data_fmt, num_fmt)");
    const auto construct_image =
        body.find("std::tuple{tsharp, image_desc}", compatibility_check);
    ASSERT_NE(compatibility_check, std::string::npos);
    ASSERT_NE(construct_image, std::string::npos);
    EXPECT_LT(compatibility_check, construct_image);
}

TEST(DynamicImageDescriptorTable, ReadsEachEntryFromTheCoherentFlatSnapshot) {
    FlatDescriptorInfo info{.flattened_ud_buf = std::vector<u32>(40)};
    const auto first = MakeImage(0x10000, AmdGpu::ImageType::Color2D,
                                 AmdGpu::NumberFormat::Unorm);
    const auto second = MakeImage(0x20000, AmdGpu::ImageType::Color2D,
                                  AmdGpu::NumberFormat::Unorm);
    std::memcpy(info.flattened_ud_buf.data() + 4, &first, sizeof(first));
    std::memcpy(info.flattened_ud_buf.data() + 12, &second, sizeof(second));

    const Shader::ImageResource resource{
        .sharp_idx = 4,
        .dynamic_descriptor_count = 2,
        .dynamic_descriptor_stride_dw = 8,
        .is_r128 = false,
        .requires_dynamic_oob_guard = true,
    };

    EXPECT_EQ(resource.NumBindings(info), 2U);
    EXPECT_EQ(resource.ReadSharpUnchecked(info, 0).Address(), first.Address());
    EXPECT_EQ(resource.ReadSharpUnchecked(info, 1).Address(), second.Address());
}

TEST(DynamicImageDescriptorTable, RequiresOneCompatibleVulkanImageTypeAndNumericClass) {
    FlatDescriptorInfo info{.flattened_ud_buf = std::vector<u32>(48)};
    const auto model = MakeImage(0x10000, AmdGpu::ImageType::Color2D,
                                 AmdGpu::NumberFormat::Unorm);
    const auto compatible = MakeImage(0x20000, AmdGpu::ImageType::Color2D,
                                      AmdGpu::NumberFormat::Snorm);
    const auto wrong_type = MakeImage(0x30000, AmdGpu::ImageType::Color3D,
                                      AmdGpu::NumberFormat::Unorm);
    const auto wrong_class = MakeImage(0x40000, AmdGpu::ImageType::Color2D,
                                       AmdGpu::NumberFormat::Uint);
    std::memcpy(info.flattened_ud_buf.data() + 4, &model, sizeof(model));
    std::memcpy(info.flattened_ud_buf.data() + 12, &compatible, sizeof(compatible));
    std::memcpy(info.flattened_ud_buf.data() + 20, &wrong_type, sizeof(wrong_type));
    std::memcpy(info.flattened_ud_buf.data() + 28, &wrong_class, sizeof(wrong_class));

    const Shader::ImageResource resource{
        .sharp_idx = 4,
        .dynamic_descriptor_count = 4,
        .dynamic_descriptor_stride_dw = 8,
        .is_r128 = false,
        .requires_dynamic_oob_guard = true,
    };

    EXPECT_TRUE(resource.IsTableEntryCompatible(info, 0));
    EXPECT_TRUE(resource.IsTableEntryCompatible(info, 1));
    EXPECT_FALSE(resource.IsTableEntryCompatible(info, 2));
    EXPECT_FALSE(resource.IsTableEntryCompatible(info, 3));
}

TEST(DynamicImageDescriptorTable, RejectsOppositeIntegerSignedness) {
    FlatDescriptorInfo info{.flattened_ud_buf = std::vector<u32>(24)};
    const auto unsigned_model = MakeImage(0x10000, AmdGpu::ImageType::Color2D,
                                          AmdGpu::NumberFormat::Uint);
    const auto signed_candidate = MakeImage(0x20000, AmdGpu::ImageType::Color2D,
                                            AmdGpu::NumberFormat::Sint);
    std::memcpy(info.flattened_ud_buf.data() + 4, &unsigned_model, sizeof(unsigned_model));
    std::memcpy(info.flattened_ud_buf.data() + 12, &signed_candidate, sizeof(signed_candidate));

    const Shader::ImageResource resource{
        .sharp_idx = 4,
        .dynamic_descriptor_count = 2,
        .dynamic_descriptor_stride_dw = 8,
        .is_r128 = false,
        .requires_dynamic_oob_guard = true,
    };

    EXPECT_TRUE(resource.IsTableEntryCompatible(info, 0));
    EXPECT_FALSE(resource.IsTableEntryCompatible(info, 1));
}

TEST(DynamicImageDescriptorTable, RejectsFmaskEntriesFromOrdinarySampledImageArrays) {
    FlatDescriptorInfo info{.flattened_ud_buf = std::vector<u32>(24)};
    const auto model = MakeImage(0x10000, AmdGpu::ImageType::Color2D,
                                 AmdGpu::NumberFormat::Unorm);
    auto fmask = MakeImage(0x20000, AmdGpu::ImageType::Color2D,
                           AmdGpu::NumberFormat::Unorm);
    fmask.data_format = static_cast<u64>(AmdGpu::DataFormat::FormatFmask8_S2_F1);
    std::memcpy(info.flattened_ud_buf.data() + 4, &model, sizeof(model));
    std::memcpy(info.flattened_ud_buf.data() + 12, &fmask, sizeof(fmask));

    const Shader::ImageResource resource{
        .sharp_idx = 4,
        .dynamic_descriptor_count = 2,
        .dynamic_descriptor_stride_dw = 8,
        .is_r128 = false,
        .requires_dynamic_oob_guard = true,
    };

    EXPECT_TRUE(resource.IsTableEntryCompatible(info, 0));
    EXPECT_FALSE(resource.IsTableEntryCompatible(info, 1));
}

TEST(DynamicImageDescriptorTable, StaticAndDynamicResourcesNeverAliasOneBinding) {
    const Shader::ImageResource static_image{
        .sharp_idx = 18,
        .is_r128 = true,
    };
    const Shader::ImageResource dynamic_image{
        .sharp_idx = 18,
        .dynamic_descriptor_count = 64,
        .dynamic_descriptor_stride_dw = 8,
        .is_r128 = true,
        .requires_dynamic_oob_guard = true,
    };
    auto different_stride = dynamic_image;
    different_stride.dynamic_descriptor_stride_dw = 16;

    EXPECT_FALSE(Shader::CanAliasImageResources(static_image, dynamic_image));
    EXPECT_TRUE(Shader::CanAliasImageResources(dynamic_image, dynamic_image));
    EXPECT_FALSE(Shader::CanAliasImageResources(dynamic_image, different_stride));
}

TEST(SamplerResourceAliasing, AnisotropySpecializationKeepsItsAssociatedImage) {
    const Shader::SamplerResource first{
        .sharp_idx = 12,
        .is_inline_sampler = false,
        .associated_image = 1,
        .disable_aniso = true,
    };
    auto different_image = first;
    different_image.associated_image = 2;

    EXPECT_TRUE(Shader::CanAliasSamplerResources(first, first));
    EXPECT_FALSE(Shader::CanAliasSamplerResources(first, different_image));

    auto ordinary_first = first;
    auto ordinary_second = different_image;
    ordinary_first.disable_aniso = false;
    ordinary_second.disable_aniso = false;
    EXPECT_TRUE(Shader::CanAliasSamplerResources(ordinary_first, ordinary_second));
}

TEST(SamplerResourceAliasing, PreservesEverySupportedAssociatedImageIndex) {
    Shader::SamplerResource sampler{
        .sharp_idx = 12,
        .is_inline_sampler = false,
        .associated_image = Shader::NUM_IMAGES - 1,
        .disable_aniso = true,
    };

    EXPECT_EQ(sampler.associated_image, Shader::NUM_IMAGES - 1);
}

TEST(DynamicImageDescriptorTable, RequiresMatchingCompileTimeSamplingSemantics) {
    FlatDescriptorInfo info{.flattened_ud_buf = std::vector<u32>(40)};
    const auto model = MakeImage(0x10000, AmdGpu::ImageType::Color2DArray,
                                 AmdGpu::NumberFormat::Unorm);
    const auto cube = MakeImage(0x20000, AmdGpu::ImageType::Cube,
                                AmdGpu::NumberFormat::Unorm);
    const auto converted = MakeImage(0x30000, AmdGpu::ImageType::Color2DArray,
                                     AmdGpu::NumberFormat::Uscaled);
    const auto srgb = MakeImage(0x40000, AmdGpu::ImageType::Color2DArray,
                                AmdGpu::NumberFormat::Srgb);
    std::memcpy(info.flattened_ud_buf.data() + 4, &model, sizeof(model));
    std::memcpy(info.flattened_ud_buf.data() + 12, &cube, sizeof(cube));
    std::memcpy(info.flattened_ud_buf.data() + 20, &converted, sizeof(converted));
    std::memcpy(info.flattened_ud_buf.data() + 28, &srgb, sizeof(srgb));

    const Shader::ImageResource resource{
        .sharp_idx = 4,
        .dynamic_descriptor_count = 4,
        .dynamic_descriptor_stride_dw = 8,
        .is_array = true,
        .is_r128 = false,
        .requires_dynamic_oob_guard = true,
    };

    EXPECT_TRUE(resource.IsTableEntryCompatible(info, 0));
    EXPECT_FALSE(resource.IsTableEntryCompatible(info, 1));
    EXPECT_FALSE(resource.IsTableEntryCompatible(info, 2));
    EXPECT_FALSE(resource.IsTableEntryCompatible(info, 3));
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
