// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <concepts>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "shader_recompiler/resource.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/texture_cache/image_binding_policy.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/types.h"
#include "video_core/renderer_vulkan/vk_resource_binding_limits.h"

namespace {

static_assert(Shader::NUM_IMAGE_BINDINGS >= Shader::NUM_IMAGES + 64,
              "A maximum-size dynamic image table must fit beside ordinary image resources");

using VideoCore::SubresourceExtent;

static_assert(std::equality_comparable<SubresourceExtent>);
static_assert(!std::totally_ordered<SubresourceExtent>);
struct DynamicImageTableInfo {
    std::array<u32, 2> flattened_ud_buf{};

    template <typename T>
    T ReadUdSharp(const u32 sharp_idx) const noexcept {
        T value{};
        std::memcpy(&value, flattened_ud_buf.data() + sharp_idx, sizeof(value));
        return value;
    }
};

std::string ReadText(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

TEST(NumberConversion, UnsupportedSnormNzStorageWidthsNeedNoExtraConversion) {
    EXPECT_EQ(AmdGpu::MapNumberConversion(AmdGpu::NumberFormat::SnormNz,
                                          AmdGpu::DataFormat::Format8_8_8_8),
              AmdGpu::NumberConversion::Sint8ToSnormNz);
    EXPECT_EQ(AmdGpu::MapNumberConversion(AmdGpu::NumberFormat::SnormNz,
                                          AmdGpu::DataFormat::Format16_16_16_16),
              AmdGpu::NumberConversion::Sint16ToSnormNz);
    EXPECT_EQ(AmdGpu::MapNumberConversion(AmdGpu::NumberFormat::SnormNz,
                                          AmdGpu::DataFormat::Format32_32_32),
              AmdGpu::NumberConversion::None);
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

TEST(VulkanDeviceFeatures, EnablesSupportedIndirectFirstInstance) {
    const auto source = ReadText(SHADPS4_VK_INSTANCE_SOURCE_PATH);

    EXPECT_NE(source.find(".drawIndirectFirstInstance = features.drawIndirectFirstInstance"),
              std::string::npos);
}

TEST(SubresourceExtent, ExpandsEachDimensionIndependently) {
    constexpr SubresourceExtent available{.levels = 3, .layers = 1};
    constexpr SubresourceExtent requested{.levels = 1, .layers = 6};

    EXPECT_EQ(available.ExpandedToFit(requested), (SubresourceExtent{.levels = 3, .layers = 6}));
}

TEST(SubresourceExtent, ExpansionIsIndependentOfOperandOrder) {
    constexpr SubresourceExtent first{.levels = 3, .layers = 1};
    constexpr SubresourceExtent second{.levels = 1, .layers = 6};

    EXPECT_EQ(first.ExpandedToFit(second), second.ExpandedToFit(first));
}

TEST(TextureBindingBookkeeping, InvalidBindingsAppendAllParallelMetadata) {
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
    EXPECT_NE(helper.find("image_bindings"), std::string::npos);
    EXPECT_NE(helper.find(".emplace_back"), std::string::npos);
    EXPECT_NE(helper.find("image_descriptor_array_sizes.push_back(descriptor_count)"),
              std::string::npos);
    EXPECT_NE(helper.find("image_native_extent_requirements.emplace_back(false, false)"),
              std::string::npos);

    size_t call_count = 0;
    size_t call = helper_end;
    while ((call = body.find("append_invalid_image_binding(image_desc.is_written, num_bindings);",
                             call)) !=
           std::string::npos) {
        ++call_count;
        ++call;
    }
    EXPECT_EQ(call_count, 1U);
}

TEST(TextureBindingBookkeeping, InvalidBindingsRetainShaderDeclaredStorageType) {
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

    EXPECT_NE(helper.find("bool is_storage"), std::string::npos);
    EXPECT_NE(helper.find("desc.type = is_storage"), std::string::npos);
    EXPECT_NE(helper.find("? VideoCore::TextureCache::BindingType::Storage"), std::string::npos);
    EXPECT_NE(helper.find(": VideoCore::TextureCache::BindingType::Texture"), std::string::npos);

    size_t typed_call_count = 0;
    size_t call = helper_end;
    while ((call = body.find("append_invalid_image_binding(image_desc.is_written, num_bindings);",
                             call)) !=
           std::string::npos) {
        ++typed_call_count;
        ++call;
    }
    EXPECT_EQ(typed_call_count, 1U);
    EXPECT_EQ(body.find("append_invalid_image_binding();", helper_end), std::string::npos);
}

TEST(TextureBindingBookkeeping, DescriptorArraysConsumeOneUnifiedBinding) {
    const auto rasterizer = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto compute_pipeline = ReadText(SHADPS4_COMPUTE_PIPELINE_SOURCE_PATH);
    const auto graphics_pipeline = ReadText(SHADPS4_GRAPHICS_PIPELINE_SOURCE_PATH);
    const auto spirv = ReadText(SHADPS4_SPIRV_EMIT_CONTEXT_SOURCE_PATH);

    const auto rasterizer_begin =
        rasterizer.find("for (u32 array_size : image_descriptor_array_sizes)");
    const auto rasterizer_end = rasterizer.find("for (const auto& sampler", rasterizer_begin);
    ASSERT_NE(rasterizer_begin, std::string::npos);
    ASSERT_NE(rasterizer_end, std::string::npos);
    const auto rasterizer_array_writes =
        rasterizer.substr(rasterizer_begin, rasterizer_end - rasterizer_begin);
    EXPECT_NE(rasterizer_array_writes.find("set_write.descriptorCount = array_size"),
              std::string::npos);
    EXPECT_NE(rasterizer_array_writes.find("binding.unified++"), std::string::npos);
    EXPECT_EQ(rasterizer_array_writes.find("binding.unified += array_size"), std::string::npos);

    for (const auto* pipeline_source : {&compute_pipeline, &graphics_pipeline}) {
        const auto image_loop = pipeline_source->find("for (const auto& image");
        ASSERT_NE(image_loop, std::string::npos);
        const auto sampler_loop = pipeline_source->find("for (const auto& sampler", image_loop);
        ASSERT_NE(sampler_loop, std::string::npos);
        const auto image_bindings = pipeline_source->substr(image_loop, sampler_loop - image_loop);
        EXPECT_NE(image_bindings.find(".descriptorCount = num_bindings"), std::string::npos);
        EXPECT_NE(image_bindings.find("binding++"), std::string::npos);
        EXPECT_EQ(image_bindings.find("binding += num_bindings"), std::string::npos);
    }

    const auto spirv_begin = spirv.find("void EmitContext::DefineImagesAndSamplers()");
    const auto spirv_end = spirv.find("if (std::ranges::any_of", spirv_begin);
    ASSERT_NE(spirv_begin, std::string::npos);
    ASSERT_NE(spirv_end, std::string::npos);
    const auto spirv_images = spirv.substr(spirv_begin, spirv_end - spirv_begin);
    EXPECT_NE(spirv_images.find("binding.unified++"), std::string::npos);
    EXPECT_EQ(spirv_images.find("binding.unified += num_bindings"), std::string::npos);
}

TEST(TextureBindingBookkeeping, InvalidDynamicTablesUpdateEveryDeclaredArrayElement) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto function_begin = source.find("void Rasterizer::BindTextures(");
    const auto function_end = source.find("RenderState Rasterizer::BeginRendering(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto helper_begin = body.find("const auto append_invalid_image_binding");
    const auto helper_end = body.find("};", helper_begin);
    ASSERT_NE(helper_begin, std::string::npos);
    ASSERT_NE(helper_end, std::string::npos);
    const auto helper = body.substr(helper_begin, helper_end - helper_begin);
    EXPECT_NE(helper.find("u32 descriptor_count"), std::string::npos);
    EXPECT_NE(helper.find("for (u32 index = 0; index < descriptor_count; ++index)"),
              std::string::npos);
    EXPECT_NE(helper.find("image_descriptor_array_sizes.push_back(descriptor_count)"),
              std::string::npos);
    EXPECT_NE(body.find("const u32 num_bindings = image_desc.NumBindings(stage)"),
              std::string::npos);
    EXPECT_NE(body.find("append_invalid_image_binding(image_desc.is_written, num_bindings)"),
              std::string::npos);
}

TEST(TextureBindingBookkeeping, PushDescriptorEligibilityUsesTotalDescriptorCount) {
    const auto compute_pipeline = ReadText(SHADPS4_COMPUTE_PIPELINE_SOURCE_PATH);
    const auto graphics_pipeline = ReadText(SHADPS4_GRAPHICS_PIPELINE_SOURCE_PATH);
    for (const auto* pipeline_source : {&compute_pipeline, &graphics_pipeline}) {
        EXPECT_NE(pipeline_source->find("u32 descriptor_count{}"), std::string::npos);
        EXPECT_NE(pipeline_source->find("descriptor_count += num_bindings"), std::string::npos);
        EXPECT_NE(pipeline_source->find("uses_push_descriptors =\n        descriptor_count <= instance.MaxPushDescriptors()"),
                  std::string::npos);
        EXPECT_EQ(pipeline_source->find("uses_push_descriptors = binding < instance.MaxPushDescriptors()"),
                  std::string::npos);
    }
}

TEST(SamplerDescriptor, RejectsReservedEnumValuesBeforeVulkanConversion) {
    AmdGpu::Sampler sampler{};
    EXPECT_TRUE(sampler.Valid());

    sampler.max_aniso.Assign(static_cast<AmdGpu::AnisoRatio>(5));
    EXPECT_FALSE(sampler.Valid());
    sampler = {};

    sampler.filter_mode.Assign(static_cast<AmdGpu::FilterMode>(3));
    EXPECT_FALSE(sampler.Valid());
    sampler = {};

    sampler.mip_filter.Assign(static_cast<AmdGpu::MipFilter>(3));
    EXPECT_FALSE(sampler.Valid());

    sampler = {};
    sampler.min_lod.Assign(0x5f8);
    sampler.max_lod.Assign(0x29f);
    EXPECT_FALSE(sampler.Valid());

    sampler.max_lod.Assign(0x5f8);
    EXPECT_TRUE(sampler.Valid());
}

TEST(DynamicImageTable, ReadsR128DescriptorAtRuntimeTableIndex) {
    alignas(16) std::array<std::byte, 3 * 32> table{};
    AmdGpu::Image first = AmdGpu::Image::Null(false);
    first.base_address = 0x1234;
    first.type = static_cast<u64>(AmdGpu::ImageType::Cube);
    first.width = 511;
    AmdGpu::Image second = first;
    second.base_address = 0x5678;
    second.width = 255;
    std::memcpy(table.data() + 16, &first, sizeof(u128));
    std::memcpy(table.data() + 48, &second, sizeof(u128));

    DynamicImageTableInfo info{};
    const VAddr table_address = reinterpret_cast<VAddr>(table.data());
    std::memcpy(info.flattened_ud_buf.data(), &table_address, sizeof(table_address));
    const Shader::ImageResource resource{
        .sharp_idx = 0,
        .is_r128 = true,
        .is_dynamic_table = true,
        .dynamic_table_count = 3,
        .dynamic_table_offset_dw = 4,
        .dynamic_table_stride_dw = 8,
    };

    const auto image = resource.GetSharpIfMapped(
        info, 1, [](const VAddr address, void* const destination, const u64 size) {
            std::memcpy(destination, reinterpret_cast<const void*>(address), size);
            return true;
        });
    EXPECT_EQ(image.Address(), second.Address());
    EXPECT_EQ(image.width, second.width);
    EXPECT_EQ(image.pitch, second.width);
    EXPECT_EQ(resource.NumBindings(info), 3u);
}

TEST(DynamicImageTable, ReadsFullDescriptorIncludingPitchAtRuntimeTableIndex) {
    alignas(16) std::array<std::byte, 2 * sizeof(AmdGpu::Image)> table{};
    AmdGpu::Image image = AmdGpu::Image::Null(false);
    image.base_address = 0x2345;
    image.width = 511;
    image.pitch = 767;
    std::memcpy(table.data() + sizeof(AmdGpu::Image), &image, sizeof(image));

    DynamicImageTableInfo info{};
    const VAddr table_address = reinterpret_cast<VAddr>(table.data());
    std::memcpy(info.flattened_ud_buf.data(), &table_address, sizeof(table_address));
    const Shader::ImageResource resource{
        .sharp_idx = 0,
        .is_r128 = false,
        .is_dynamic_table = true,
        .dynamic_table_count = 2,
        .dynamic_table_offset_dw = 0,
        .dynamic_table_stride_dw = sizeof(AmdGpu::Image) / sizeof(u32),
    };

    const auto loaded = resource.GetSharpIfMapped(
        info, 1, [](const VAddr address, void* const destination, const u64 size) {
            std::memcpy(destination, reinterpret_cast<const void*>(address), size);
            return true;
        });
    EXPECT_EQ(loaded.Address(), image.Address());
    EXPECT_EQ(loaded.width, image.width);
    EXPECT_EQ(loaded.pitch, image.pitch);
}

TEST(DynamicImageTable, CompilationUsesCapturedDescriptorWithoutRereadingGuestTable) {
    DynamicImageTableInfo info{};
    constexpr VAddr stale_table_address = 0xFFFFFFFFFFFFFFFFULL;
    std::memcpy(info.flattened_ud_buf.data(), &stale_table_address,
                sizeof(stale_table_address));

    AmdGpu::Image captured = AmdGpu::Image::Null(false);
    captured.base_address = 0x3456;
    captured.width = 511;
    captured.pitch = 767;
    const Shader::ImageResource resource{
        .sharp_idx = 0,
        .is_dynamic_table = true,
        .dynamic_table_count = 64,
        .dynamic_table_offset_dw = 3620,
        .dynamic_table_stride_dw = 8,
        .dynamic_table_template = captured,
    };

    const auto loaded = resource.GetSharp(info);
    EXPECT_EQ(loaded.Address(), captured.Address());
    EXPECT_EQ(loaded.width, captured.width);
    EXPECT_EQ(loaded.pitch, captured.pitch);
}

TEST(DynamicImageTable, RejectsUnmappedDescriptorBeforeDereferencingGuestMemory) {
    DynamicImageTableInfo info{};
    constexpr VAddr table_address = 0x12345000;
    std::memcpy(info.flattened_ud_buf.data(), &table_address, sizeof(table_address));
    const Shader::ImageResource resource{
        .sharp_idx = 0,
        .is_r128 = false,
        .is_dynamic_table = true,
        .dynamic_table_count = 2,
        .dynamic_table_offset_dw = 4,
        .dynamic_table_stride_dw = sizeof(AmdGpu::Image) / sizeof(u32),
    };

    bool queried = false;
    const auto loaded = resource.GetSharpIfMapped(
        info, 1, [&](const VAddr address, void* const destination, const u64 size) {
            queried = true;
            EXPECT_EQ(address, table_address + 16 + sizeof(AmdGpu::Image));
            EXPECT_NE(destination, nullptr);
            EXPECT_EQ(size, sizeof(AmdGpu::Image));
            return false;
        });

    EXPECT_TRUE(queried);
    EXPECT_EQ(loaded.Address(), 0U);
}

TEST(DynamicImageTable, RejectsEveryUnmappedRuntimeImageBeforeTextureCacheAdmission) {
    AmdGpu::Image image = AmdGpu::Image::Null(false);
    image.base_address = 0x242b0fc00ULL;
    image.width = 63;
    image.height = 63;
    image.pitch = 63;
    constexpr VideoCore::ImageBindingLimits limits{
        .max_dimension_1d = 16384,
        .max_dimension_2d = 16384,
        .max_dimension_3d = 2048,
        .max_dimension_cube = 16384,
        .max_array_layers = 2048,
        .max_guest_resource_size = 8ULL * 1024 * 1024 * 1024,
    };

    bool mapping_queried = false;
    EXPECT_FALSE(VideoCore::IsGuestImageBindingUsable(
        image.Address(), image.pitch, image.width, VideoCore::ImageBindingKind::TwoDimensional,
        VideoCore::ImageBindingExtent{.width = 64, .height = 64, .depth = 1, .layers = 1},
        0x4000, limits, [&](VAddr address, u64 size) {
            mapping_queried = true;
            EXPECT_EQ(address, image.Address());
            EXPECT_EQ(size, 1U);
            return false;
        }));
    EXPECT_TRUE(mapping_queried);
}

TEST(TextureBindingBookkeeping, ValidatesEveryDynamicTableEntryBeforeTextureCacheAdmission) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto function_begin = source.find("void Rasterizer::BindTextures(");
    const auto function_end = source.find("RenderState Rasterizer::BeginRendering(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);
    const auto loop_begin = body.find("for (auto i = 0; i < num_bindings; i++)");
    const auto loop_end = body.find("image_descriptor_array_sizes.push_back", loop_begin);
    ASSERT_NE(loop_begin, std::string::npos);
    ASSERT_NE(loop_end, std::string::npos);
    const auto loop = body.substr(loop_begin, loop_end - loop_begin);
    const auto validation = loop.find("is_image_binding_usable(binding_tsharp, image_desc,");
    const auto admission = loop.find("texture_cache.FindImage(desc)");
    ASSERT_NE(validation, std::string::npos);
    ASSERT_NE(admission, std::string::npos);
    EXPECT_LT(validation, admission);
    EXPECT_NE(loop.find("desc.type = image_desc.is_written"), std::string::npos);
}

TEST(TextureBindingBookkeeping, DynamicTableViewsMatchTheCompiledDescriptorTemplate) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto function_begin = source.find("void Rasterizer::BindTextures(");
    const auto function_end = source.find("RenderState Rasterizer::BeginRendering(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto template_sharp = body.find("const auto template_tsharp = image_desc.GetSharp(stage)");
    const auto template_view = body.find("template_tsharp.GetViewType(image_desc.is_array)");
    const auto runtime_first_view =
        body.find("const auto expected_view_type = tsharp.GetViewType(image_desc.is_array)");
    const auto entry_check = body.find(
        "binding_tsharp.GetViewType(image_desc.is_array) != expected_view_type");
    const auto template_integer =
        body.find("AmdGpu::IsInteger(template_tsharp.GetNumberFmt())");
    const auto runtime_first_integer = body.find("AmdGpu::IsInteger(num_fmt)");

    EXPECT_NE(template_sharp, std::string::npos);
    EXPECT_NE(template_view, std::string::npos);
    EXPECT_EQ(runtime_first_view, std::string::npos);
    EXPECT_NE(entry_check, std::string::npos);
    EXPECT_NE(template_integer, std::string::npos);
    EXPECT_EQ(runtime_first_integer, std::string::npos);
}

TEST(DynamicImageTable, ReusedDescriptorIndexesReceiveUniqueNonUniformIds) {
    const auto source = ReadText(SHADPS4_SPIRV_EMIT_IMAGE_SOURCE_PATH);
    const auto function_begin = source.find("Id EmitImageSampleExplicitLodDynamic(");
    const auto function_end = source.find("Id EmitImageSampleDrefImplicitLod(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    EXPECT_NE(body.find("OpIAdd(ctx.U32[1], descriptor_index, ctx.u32_zero_value)"),
              std::string::npos);
    EXPECT_NE(body.find("Decorate(nonuniform_index, spv::Decoration::NonUniform)"),
              std::string::npos);
    EXPECT_EQ(body.find("Decorate(descriptor_index, spv::Decoration::NonUniform)"),
              std::string::npos);
}

TEST(TextureBindingBookkeeping, PipelineImageStorageGrowsForDynamicDescriptorTables) {
    const auto header = ReadText(SHADPS4_RASTERIZER_HEADER_PATH);
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    EXPECT_NE(header.find("std::vector<vk::DescriptorImageInfo> image_infos"), std::string::npos);
    EXPECT_NE(header.find("std::vector<vk::DescriptorBufferInfo> buffer_infos"), std::string::npos);
    EXPECT_NE(header.find("std::vector<VideoCore::ImageId> bound_images"), std::string::npos);
    EXPECT_NE(header.find("std::vector<ImageBindingInfo> image_bindings"), std::string::npos);

    const auto bind_begin = source.find("bool Rasterizer::BindResources(");
    const auto bind_end = source.find("void Rasterizer::PrepareAddr64PointerResidency(", bind_begin);
    ASSERT_NE(bind_begin, std::string::npos);
    ASSERT_NE(bind_end, std::string::npos);
    const auto bind = source.substr(bind_begin, bind_end - bind_begin);
    const auto reserve = bind.find("image_infos.reserve(required_image_infos)");
    const auto buffer_reserve = bind.find("buffer_infos.reserve(required_buffer_infos)");
    const auto texture_binding = bind.find("BindTextures(*stage, binding)");
    ASSERT_NE(reserve, std::string::npos);
    ASSERT_NE(buffer_reserve, std::string::npos);
    ASSERT_NE(texture_binding, std::string::npos);
    EXPECT_LT(reserve, texture_binding);
    EXPECT_LT(buffer_reserve, texture_binding);
}

TEST(ImageBindingFormat, RejectsKnownEnumsWithoutAVulkanSurfaceFormat) {
    EXPECT_FALSE(Vulkan::LiverpoolToVK::IsSurfaceFormatKnown(
        static_cast<AmdGpu::DataFormat>(20), static_cast<AmdGpu::NumberFormat>(4)));
    EXPECT_TRUE(Vulkan::LiverpoolToVK::IsSurfaceFormatKnown(
        AmdGpu::DataFormat::Format8_8_8_8, AmdGpu::NumberFormat::Unorm));
}

TEST(TextureBindingBookkeeping, ChecksFormatPairsBeforeAssertingVulkanConversion) {
    const auto rasterizer = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto pipeline_cache = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    for (const auto* source : {&rasterizer, &pipeline_cache}) {
        const auto known = source->find("IsImageBindingFormatKnown(");
        const auto conversion = source->find("DescribeImageBindingFormat(");
        ASSERT_NE(known, std::string::npos);
        ASSERT_NE(conversion, std::string::npos);
        EXPECT_LT(known, conversion);
    }
}

TEST(ShaderPermutationMetadata, RetainsTheInfoUsedToCompileEachModule) {
    const auto header = ReadText(SHADPS4_PIPELINE_CACHE_HEADER_PATH);
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    const auto module_begin = header.find("struct Module {");
    const auto module_end = header.find("\n    };", module_begin);
    ASSERT_NE(module_begin, std::string::npos);
    ASSERT_NE(module_end, std::string::npos);
    const auto module = header.substr(module_begin, module_end - module_begin);
    EXPECT_NE(module.find("std::unique_ptr<Shader::Info> info"), std::string::npos);

    const auto get_program_begin = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto get_program_end = source.find("Shader::SrtMemorySynchronizer", get_program_begin);
    ASSERT_NE(get_program_begin, std::string::npos);
    ASSERT_NE(get_program_end, std::string::npos);
    const auto get_program = source.substr(get_program_begin, get_program_end - get_program_begin);
    EXPECT_NE(get_program.find("module.GetInfo(program->info)"), std::string::npos);
    EXPECT_NE(get_program.find("std::make_unique<Shader::Info>"), std::string::npos);
    EXPECT_NE(get_program.find("return std::make_tuple(module_info"), std::string::npos);
}

TEST(ShaderPermutationMetadata, CachePreloadRetainsEachPermutationsDeserializedInfo) {
    const auto source = ReadText(SHADPS4_PIPELINE_SERIALIZATION_SOURCE_PATH);
    const auto load_begin = source.find("bool PipelineCache::LoadPipelineStage(");
    const auto load_end = source.find("void PipelineCache::WarmUp()", load_begin);
    ASSERT_NE(load_begin, std::string::npos);
    ASSERT_NE(load_end, std::string::npos);
    const auto load = source.substr(load_begin, load_end - load_begin);

    EXPECT_NE(load.find("std::make_unique<Shader::Info>()"), std::string::npos);
    EXPECT_NE(load.find("LoadShaderMeta(ar, *loaded_info"), std::string::npos);
    EXPECT_NE(load.find("std::move(loaded_info)"), std::string::npos);
    EXPECT_NE(load.find("GetInfo(it_pgm.value()->info)"), std::string::npos);
    EXPECT_EQ(load.find("LoadShaderMeta(ar, program->info"), std::string::npos);
    EXPECT_EQ(load.find("infos[stage] = &it_pgm.value()->info"), std::string::npos);
}

TEST(ShaderPermutationMetadata, SparseCachePreloadSlotsAreNotRuntimePermutations) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto get_program_begin = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto get_program_end = source.find("Shader::SrtMemorySynchronizer", get_program_begin);
    ASSERT_NE(get_program_begin, std::string::npos);
    ASSERT_NE(get_program_end, std::string::npos);
    const auto get_program =
        source.substr(get_program_begin, get_program_end - get_program_begin);

    const auto loop = get_program.find("for (size_t perm_idx = 0;");
    const auto skip_empty = get_program.find("if (!module.module)", loop);
    const auto read_info = get_program.find("module.GetInfo(program->info)", loop);
    ASSERT_NE(loop, std::string::npos);
    ASSERT_NE(skip_empty, std::string::npos);
    ASSERT_NE(read_info, std::string::npos);
    EXPECT_LT(skip_empty, read_info);
}

TEST(ShaderPermutationMetadata, BindingValidityTransitionsRequireANewPermutation) {
    const auto source = ReadText(SHADPS4_SPECIALIZATION_HEADER_PATH);
    const auto equality_begin = source.find("bool operator==(const StageSpecialization& other)");
    const auto equality_end = source.find("void Serialize", equality_begin);
    ASSERT_NE(equality_begin, std::string::npos);
    ASSERT_NE(equality_end, std::string::npos);
    const auto equality = source.substr(equality_begin, equality_end - equality_begin);

    EXPECT_NE(equality.find("bitset[binding] != other.bitset[binding]"), std::string::npos);
    EXPECT_NE(equality.find("if (bitset[binding] &&"), std::string::npos);
}

TEST(TextureBindingBookkeeping, ProductionBindingHasNoTitleSpecificShaderDiagnostics) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto function_begin = source.find("void Rasterizer::BindTextures(");
    const auto function_end = source.find("RenderState Rasterizer::BeginRendering(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    EXPECT_EQ(body.find("0xfb974c0e"), std::string::npos);
    const auto rejection = body.find("Rejecting unusable T#");
    ASSERT_NE(rejection, std::string::npos);
    const auto log_call = body.rfind("LOG_", rejection);
    ASSERT_NE(log_call, std::string::npos);
    EXPECT_EQ(body.substr(log_call, 9), "LOG_DEBUG");
}

TEST(TextureBindingBookkeeping, UsesActualRuntimeMappingsForImageAdmission) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto function_begin = source.find("void Rasterizer::BindTextures(");
    const auto function_end = source.find("RenderState Rasterizer::BeginRendering(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    EXPECT_NE(body.find("return ReadMappedMemory(address, destination, size);"),
              std::string::npos);
    EXPECT_EQ(body.find("return memory->IsValidGpuMapping(address, size);"),
              std::string::npos);
}

TEST(TextureBindingBookkeeping, InvalidSamplersFallBackBeforeSamplerCreation) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto function_begin = source.find("void Rasterizer::BindTextures(");
    const auto function_end = source.find("RenderState Rasterizer::BeginRendering(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto validity_check = body.find("if (!ssharp.Valid())");
    const auto sampler_creation = body.find("texture_cache.GetSampler(ssharp", validity_check);
    ASSERT_NE(validity_check, std::string::npos);
    ASSERT_NE(sampler_creation, std::string::npos);
    EXPECT_LT(validity_check, sampler_creation);
}

TEST(TextureBindingBookkeeping, FirstStorageBindingForcesSharedImageGeneralLayout) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto function_begin = source.find("void Rasterizer::BindTextures(");
    const auto function_end = source.find("RenderState Rasterizer::BeginRendering(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto force_general =
        body.find("image->binding.force_general |= image_desc.is_written");
    const auto mark_bound = body.find("image->binding.is_bound = 1u", force_general);
    ASSERT_NE(force_general, std::string::npos);
    ASSERT_NE(mark_bound, std::string::npos);
    const auto preceding_if = body.rfind("if (image->binding.is_bound)", force_general);
    const auto preceding_block_end = body.rfind('}', force_general);
    EXPECT_TRUE(preceding_if == std::string::npos || preceding_block_end > preceding_if);
    EXPECT_LT(force_general, mark_bound);
}

TEST(ImageArgumentPatching, SampledMsaaArraysUseThreeCoordinates) {
    const auto source = ReadText(SHADPS4_RESOURCE_TRACKING_SOURCE_PATH);
    const auto function_begin = source.find("void PatchImageSampleArgs(");
    const auto function_end = source.find("void PatchImageArgs(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto coordinates_begin = body.find("const IR::Value coords =");
    ASSERT_NE(coordinates_begin, std::string::npos);
    const auto array_case =
        body.find("case AmdGpu::ImageType::Color2DMsaaArray:", coordinates_begin);
    const auto three_coordinates = body.find("addr_reg = addr_reg + 3;", array_case);
    const auto default_case = body.find("default:", array_case);
    ASSERT_NE(array_case, std::string::npos);
    ASSERT_NE(three_coordinates, std::string::npos);
    ASSERT_NE(default_case, std::string::npos);
    EXPECT_LT(three_coordinates, default_case);
}

TEST(BufferAddressPatching, AddThreadIdIsNotRestrictedToComputeShaders) {
    const auto source = ReadText(SHADPS4_RESOURCE_TRACKING_SOURCE_PATH);
    const auto function_begin = source.find("IR::U32 CalculateBufferAddress(");
    const auto function_end = source.find("void PatchBufferArgs(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto add_thread_id = body.find("if (buffer.add_tid_enable)");
    const auto lane_id = body.find("ir.LaneId()", add_thread_id);
    ASSERT_NE(add_thread_id, std::string::npos);
    ASSERT_NE(lane_id, std::string::npos);
    EXPECT_EQ(body.find("info.l_stage == LogicalStage::Compute"), std::string::npos);
}

TEST(TextureMetadata, DepthOverlapExpansionRecomputesGuestLayoutBeforeAllocation) {
    const auto source = ReadText(SHADPS4_TEXTURE_CACHE_SOURCE_PATH);
    const auto function_begin = source.find("ImageId TextureCache::ResolveDepthOverlap(");
    const auto function_end = source.find("std::tuple<ImageId, int, int> TextureCache::ResolveOverlap(",
                                          function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto resource_growth = body.find("new_info.resources =");
    ASSERT_NE(resource_growth, std::string::npos);
    const auto layout_update = body.find("new_info.UpdateSize();", resource_growth);
    const auto allocation = body.find("slot_images.insert", resource_growth);
    ASSERT_NE(layout_update, std::string::npos);
    ASSERT_NE(allocation, std::string::npos);
    EXPECT_LT(layout_update, allocation);
}

} // namespace
