// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <gtest/gtest.h>

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/specialization_serialization.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/descriptor_binding_cursor.h"
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

AmdGpu::Image MakeImage(VAddr address, AmdGpu::ImageType type, AmdGpu::NumberFormat number_format) {
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
    for (const auto type :
         {AmdGpu::ImageType::Color1D, AmdGpu::ImageType::Color2D, AmdGpu::ImageType::Color2DMsaa}) {
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
    const auto function_end =
        source.find("RenderState Rasterizer::BeginRendering(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto helper_begin = body.find("const auto append_invalid_image_binding");
    ASSERT_NE(helper_begin, std::string::npos);
    const auto helper_end = body.find("};", helper_begin);
    ASSERT_NE(helper_end, std::string::npos);
    const auto helper = body.substr(helper_begin, helper_end - helper_begin);
    EXPECT_NE(helper.find("image_bindings.emplace_back"), std::string::npos);
    EXPECT_NE(helper.find("BindingType::Storage"), std::string::npos);
    EXPECT_NE(helper.find("BindingType::Texture"), std::string::npos);
    EXPECT_NE(helper.find("image_native_extent_requirements.emplace_back(false, false)"),
              std::string::npos);

    size_t call_count = 0;
    size_t call = helper_end;
    while ((call = body.find("append_invalid_image_binding(image_desc.is_written);", call)) !=
           std::string::npos) {
        ++call_count;
        ++call;
    }
    EXPECT_EQ(call_count, 3U);
    EXPECT_NE(body.find("image_descriptor_array_sizes.push_back(num_bindings)"), std::string::npos);
    const auto compatibility_check = body.find("image_desc.IsTableEntryCompatible(stage, i)");
    const auto read_sharp = body.find("image_desc.GetSharp(stage, i)");
    ASSERT_NE(compatibility_check, std::string::npos);
    ASSERT_NE(read_sharp, std::string::npos);
    EXPECT_LT(compatibility_check, read_sharp);
}

TEST(TextureBindingBookkeeping, ClipsBufferToMappedExtentBeforeCaching) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto function_begin = source.find("void Rasterizer::BindBuffers(");
    const auto function_end = source.find("void Rasterizer::BindTextures(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto address_check = body.find("memory->IsValidMapping(vsharp.base_address)");
    const auto range_bound = body.find("memory->ClampRangeSize(vsharp.base_address, binding_size)");
    const auto mapped_extent = body.find("page_manager.GetGpuMappedExtent(");
    const auto nonempty_extent = body.find("if (size != 0)", mapped_extent);
    const auto cache_lookup = body.find("buffer_cache.FindBuffer(vsharp.base_address, size)");
    ASSERT_NE(address_check, std::string::npos);
    ASSERT_NE(range_bound, std::string::npos);
    ASSERT_NE(mapped_extent, std::string::npos);
    ASSERT_NE(nonempty_extent, std::string::npos);
    ASSERT_NE(cache_lookup, std::string::npos);
    EXPECT_LT(address_check, range_bound);
    EXPECT_LT(range_bound, mapped_extent);
    EXPECT_NE(body.find("page_manager.GetGpuMappedExtent(vsharp.base_address, bounded_size)"),
              std::string::npos);
    EXPECT_LT(mapped_extent, nonempty_extent);
    EXPECT_LT(nonempty_extent, cache_lookup);
    EXPECT_EQ(body.find("page_manager.IsGpuMapped(vsharp.base_address, 1)"), std::string::npos);
}

TEST(TextureBindingBookkeeping, ScratchStorageCanGrowPastPerStageResourceCount) {
    // A valid runtime-indexed table can consume all image slots while ordinary resources and
    // samplers from the same stage still need adjacent descriptor storage.
    constexpr size_t expanded_descriptor_count = 11U + Shader::NUM_IMAGES + 2U;
    boost::container::small_vector<u32, Shader::NUM_IMAGES> descriptors;

    descriptors.reserve(expanded_descriptor_count);
    for (u32 descriptor = 0; descriptor < expanded_descriptor_count; ++descriptor) {
        descriptors.emplace_back(descriptor);
    }

    ASSERT_EQ(descriptors.size(), expanded_descriptor_count);
    EXPECT_EQ(descriptors.front(), 0U);
    EXPECT_EQ(descriptors.back(), expanded_descriptor_count - 1U);
}

TEST(TextureBindingBookkeeping, RasterizerUsesGrowableDescriptorScratchStorage) {
    const auto header = ReadText(SHADPS4_RASTERIZER_HEADER_PATH);

    EXPECT_NE(header.find("small_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES>"),
              std::string::npos);
    EXPECT_NE(header.find("small_vector<DescriptorImageLayoutBinding, Shader::NUM_IMAGES>"),
              std::string::npos);
    EXPECT_NE(header.find("small_vector<VideoCore::ImageId, Shader::NUM_IMAGES>"),
              std::string::npos);
    EXPECT_NE(header.find("small_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS>"),
              std::string::npos);
    EXPECT_NE(header.find("small_vector<ImageBindingInfo, Shader::NUM_IMAGES>"),
              std::string::npos);
}

} // namespace

TEST(DescriptorBindingCursor, MultipleArraysConsumeOneBindingEachAndAllCapacity) {
    VideoCore::DescriptorBindingCursor cursor;
    EXPECT_EQ(cursor.TryAdd(3), 0U);
    EXPECT_EQ(cursor.TryAdd(5), 1U);
    EXPECT_EQ(cursor.BindingCount(), 2U);
    EXPECT_EQ(cursor.DescriptorCount(), 8U);
}

TEST(DescriptorBindingCursor, MixedSinglesAndArraysKeepDeterministicOrigins) {
    VideoCore::DescriptorBindingCursor cursor;
    EXPECT_EQ(cursor.TryAdd(1), 0U);
    EXPECT_EQ(cursor.TryAdd(4), 1U);
    EXPECT_EQ(cursor.TryAdd(1), 2U);
    EXPECT_EQ(cursor.BindingCount(), 3U);
    EXPECT_EQ(cursor.DescriptorCount(), 6U);
}

TEST(DescriptorBindingCursor, ExactCapacityIsAllowed) {
    VideoCore::DescriptorBindingCursor cursor;
    ASSERT_TRUE(cursor.TryAdd(8).has_value());
    EXPECT_TRUE(cursor.CanUsePushDescriptors(8));
    EXPECT_FALSE(cursor.CanUsePushDescriptors(7));
}

TEST(DescriptorBindingCursor, ZeroAndOverflowFailWithoutMutation) {
    VideoCore::DescriptorBindingCursor cursor;
    EXPECT_FALSE(cursor.TryAdd(0).has_value());
    EXPECT_EQ(cursor.BindingCount(), 0U);
    EXPECT_EQ(cursor.DescriptorCount(), 0U);

    ASSERT_TRUE(cursor.TryAdd(std::numeric_limits<u32>::max()).has_value());
    EXPECT_FALSE(cursor.TryAdd(1).has_value());
    EXPECT_EQ(cursor.BindingCount(), 1U);
    EXPECT_EQ(cursor.DescriptorCount(), std::numeric_limits<u32>::max());
}

TEST(DescriptorArrayBinding, EmissionAndWritesReuseDeterministicArrayOrigins) {
    Shader::Backend::Bindings emitted;
    Shader::Backend::Bindings written;

    for (const u32 descriptor_count : {3U, 5U, 1U}) {
        const u32 expected_binding = emitted.unified;
        EXPECT_EQ(emitted.TryAddImageBinding(descriptor_count), expected_binding);
        EXPECT_EQ(written.TryAddImageBinding(descriptor_count), expected_binding);
    }
    EXPECT_EQ(emitted.unified, 3U);
    EXPECT_EQ(written, emitted);
}

TEST(DescriptorArrayBinding, MalformedAndOverflowInputsFailWithoutMutation) {
    Shader::Backend::Bindings binding;
    EXPECT_FALSE(binding.TryAddImageBinding(0).has_value());
    EXPECT_EQ(binding.unified, 0U);

    binding.unified = std::numeric_limits<u32>::max();
    EXPECT_FALSE(binding.TryAddImageBinding(1).has_value());
    EXPECT_EQ(binding.unified, std::numeric_limits<u32>::max());
}

TEST(DescriptorArrayBinding, MixedStagesKeepProductionOriginsAligned) {
    Shader::Backend::Bindings spirv;
    Shader::Backend::Bindings writes;
    VideoCore::DescriptorBindingCursor layout;

    const auto apply_stage = [](Shader::Backend::Bindings& bindings, u32 first_array,
                                u32 second_array) {
        std::array<std::optional<u32>, 5> origins{
            bindings.TryAddBufferBinding(), bindings.TryAddImageBinding(first_array),
            bindings.TryAddImageBinding(second_array), bindings.TryAddSamplerBinding(),
            bindings.TryAddBufferBinding()};
        return origins;
    };

    const auto apply_layout_stage = [](VideoCore::DescriptorBindingCursor& bindings,
                                       u32 first_array, u32 second_array) {
        std::array<std::optional<u32>, 5> origins{bindings.TryAdd(), bindings.TryAdd(first_array),
                                                  bindings.TryAdd(second_array), bindings.TryAdd(),
                                                  bindings.TryAdd()};
        return origins;
    };

    const auto spirv_origins = apply_stage(spirv, 3, 5);
    const auto layout_origins = apply_layout_stage(layout, 3, 5);
    const auto write_origins = apply_stage(writes, 3, 5);
    EXPECT_EQ(spirv_origins, layout_origins);
    EXPECT_EQ(layout_origins, write_origins);
    EXPECT_EQ(spirv_origins, (std::array<std::optional<u32>, 5>{0U, 1U, 2U, 3U, 4U}));
    EXPECT_EQ(spirv.unified, 5U);
    EXPECT_EQ(spirv.buffer, 2U);
    EXPECT_EQ(layout.DescriptorCount(), 11U);
}

TEST(DynamicImageDescriptorTable, ReadsBoundedEntriesFromTheFlattenedSnapshot) {
    FlatDescriptorInfo info{.flattened_ud_buf = std::vector<u32>(32)};
    const auto first = MakeImage(0x10000, AmdGpu::ImageType::Color2D, AmdGpu::NumberFormat::Unorm);
    const auto second = MakeImage(0x20000, AmdGpu::ImageType::Color2D, AmdGpu::NumberFormat::Snorm);
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
    EXPECT_TRUE(resource.IsTableEntryCompatible(info, 0));
    EXPECT_TRUE(resource.IsTableEntryCompatible(info, 1));
    EXPECT_FALSE(resource.IsTableEntryCompatible(info, 2));
}

TEST(DynamicImageDescriptorTable, RejectsIncompatibleTypesAndNumericClasses) {
    FlatDescriptorInfo info{.flattened_ud_buf = std::vector<u32>(40)};
    const auto model = MakeImage(0x10000, AmdGpu::ImageType::Color2D, AmdGpu::NumberFormat::Uint);
    const auto wrong_type =
        MakeImage(0x20000, AmdGpu::ImageType::Color3D, AmdGpu::NumberFormat::Uint);
    const auto wrong_class =
        MakeImage(0x30000, AmdGpu::ImageType::Color2D, AmdGpu::NumberFormat::Sint);
    std::memcpy(info.flattened_ud_buf.data() + 4, &model, sizeof(model));
    std::memcpy(info.flattened_ud_buf.data() + 12, &wrong_type, sizeof(wrong_type));
    std::memcpy(info.flattened_ud_buf.data() + 20, &wrong_class, sizeof(wrong_class));

    const Shader::ImageResource resource{
        .sharp_idx = 4,
        .dynamic_descriptor_count = 3,
        .dynamic_descriptor_stride_dw = 8,
        .is_r128 = false,
        .requires_dynamic_oob_guard = true,
    };

    EXPECT_TRUE(resource.IsTableEntryCompatible(info, 0));
    EXPECT_FALSE(resource.IsTableEntryCompatible(info, 1));
    EXPECT_FALSE(resource.IsTableEntryCompatible(info, 2));
}

TEST(DynamicImageDescriptorTable, RejectsMalformedOrOutOfSnapshotRangesBeforeReading) {
    FlatDescriptorInfo info{.flattened_ud_buf = std::vector<u32>(32)};
    const auto model = MakeImage(0x10000, AmdGpu::ImageType::Color2D, AmdGpu::NumberFormat::Unorm);
    std::memcpy(info.flattened_ud_buf.data() + 4, &model, sizeof(model));
    Shader::ImageResource resource{
        .sharp_idx = 4,
        .dynamic_descriptor_count = 4,
        .dynamic_descriptor_stride_dw = 8,
        .is_r128 = false,
        .requires_dynamic_oob_guard = true,
    };

    EXPECT_FALSE(resource.HasValidDynamicRange(info));
    EXPECT_FALSE(resource.IsTableEntryCompatible(info, 0));
    EXPECT_EQ(resource.NumBindings(info), 1U);
    EXPECT_EQ(resource.GetSharp(info, 0).Address(), 0U);

    resource.dynamic_descriptor_count = Shader::NUM_IMAGES + 1U;
    EXPECT_FALSE(resource.HasValidDynamicRange(info));
    resource.dynamic_descriptor_count = 2U;
    resource.dynamic_descriptor_stride_dw = std::numeric_limits<u32>::max();
    EXPECT_FALSE(resource.HasValidDynamicRange(info));
}

TEST(DynamicImageDescriptorTable, AcceptsExactlyBoundedReservationRange) {
    FlatDescriptorInfo info{.flattened_ud_buf = std::vector<u32>(532)};
    const Shader::ImageResource resource{
        .sharp_idx = 20,
        .dynamic_descriptor_count = Shader::NUM_IMAGES,
        .dynamic_descriptor_stride_dw = 8,
        .is_r128 = false,
        .requires_dynamic_oob_guard = true,
    };

    EXPECT_TRUE(resource.HasValidDynamicRange(info));
    EXPECT_EQ(resource.NumBindings(info), Shader::NUM_IMAGES);
}

TEST(DynamicImageDescriptorTable, StaticAndDynamicResourcesCannotAlias) {
    const Shader::ImageResource static_image{.sharp_idx = 18, .is_r128 = true};
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

TEST(SamplerResourceAliasing, KeepsTheAssociatedImageWhenAnisotropyDependsOnIt) {
    const Shader::SamplerResource first{
        .sharp_idx = 12,
        .is_inline_sampler = false,
        .associated_image = Shader::NUM_IMAGES - 1,
        .disable_aniso = true,
    };
    auto different_image = first;
    different_image.associated_image = 0;

    EXPECT_TRUE(Shader::CanAliasSamplerResources(first, first));
    EXPECT_FALSE(Shader::CanAliasSamplerResources(first, different_image));

    auto ordinary_first = first;
    auto ordinary_second = different_image;
    ordinary_first.disable_aniso = false;
    ordinary_second.disable_aniso = false;
    EXPECT_TRUE(Shader::CanAliasSamplerResources(ordinary_first, ordinary_second));
}

TEST(DynamicImageDescriptorTable, InvalidatesRawShaderMetadataCacheLayout) {
    EXPECT_EQ(Shader::ShaderBinaryVersion, 8U);
    EXPECT_EQ(Shader::ShaderMetaVersion, 8U);
}

TEST(DynamicImageDescriptorTable, SerializesSrtReservationsFieldByField) {
    const auto source = ReadText(SHADPS4_PIPELINE_SERIALIZATION_SOURCE_PATH);
    const auto serialize = source.find("void PersistentSrtInfo::Serialize");
    const auto deserialize = source.find("bool PersistentSrtInfo::Deserialize");
    ASSERT_NE(serialize, std::string::npos);
    ASSERT_NE(deserialize, std::string::npos);
    const auto writer = source.substr(serialize, deserialize - serialize);
    EXPECT_EQ(writer.find("srt.Write(this, sizeof(*this))"), std::string::npos);
    EXPECT_NE(writer.find("reservation.pointer_lo_flat_index"), std::string::npos);
    EXPECT_NE(writer.find("reservation.source_offset_dw"), std::string::npos);
    EXPECT_NE(writer.find("reservation.destination_flat_offset_dw"), std::string::npos);
    EXPECT_NE(writer.find("reservation.span_dw"), std::string::npos);

    const auto reader = source.substr(deserialize);
    EXPECT_NE(reader.find("MaxRangeReservations"), std::string::npos);
    EXPECT_NE(reader.find("ValidateRangeReservations"), std::string::npos);
}
