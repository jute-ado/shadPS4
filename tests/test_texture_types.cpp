// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "video_core/amdgpu/resource.h"
#include "video_core/descriptor_binding_cursor.h"
#include "video_core/texture_cache/image_layout_policy.h"
#include "video_core/texture_cache/types.h"

namespace {

using VideoCore::SubresourceExtent;

TEST(ImageLayoutPolicy, ClassifiesPrtThinAsMacroTiled) {
    const auto layout =
        VideoCore::ClassifyImageArrayLayout(AmdGpu::ArrayMode::ArrayPrtTiledThin1);

    EXPECT_EQ(layout.kind, VideoCore::ImageArrayLayoutKind::MacroTiled);
    EXPECT_EQ(layout.thickness, 1U);
}

TEST(ImageLayoutPolicy, ConvertsBlockCompressedMipGeometryToAndFromBlockUnits) {
    const auto blocks = VideoCore::ToBlockCompressedMipGeometry(257, 129);
    EXPECT_EQ(blocks.pitch, 65U);
    EXPECT_EQ(blocks.height, 33U);

    const auto texels = VideoCore::FromBlockCompressedMipGeometry(64, 32);
    EXPECT_EQ(texels.pitch, 256U);
    EXPECT_EQ(texels.height, 128U);
}

TEST(ImageLayoutPolicy, RestoredBlockCompressedGeometryKeepsMinimumTileExtent) {
    const auto texels = VideoCore::FromBlockCompressedMipGeometry(1, 1);
    EXPECT_EQ(texels.pitch, 32U);
    EXPECT_EQ(texels.height, 32U);
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

TEST(ComponentMapping, RejectsReservedGuestSelectorEncodings) {
    EXPECT_TRUE(AmdGpu::IdentityMapping.Valid());

    AmdGpu::CompMapping mapping = AmdGpu::IdentityMapping;
    mapping.g = static_cast<AmdGpu::CompSwizzle>(2);
    EXPECT_FALSE(mapping.Valid());
    mapping.g = static_cast<AmdGpu::CompSwizzle>(3);
    EXPECT_FALSE(mapping.Valid());
}

TEST(NumberConversion, RejectsUnsupportedPackedSnormNzDescriptorsWithoutAsserting) {
    EXPECT_TRUE(AmdGpu::IsNumberConversionSupported(AmdGpu::NumberFormat::SnormNz,
                                                    AmdGpu::DataFormat::Format8));
    EXPECT_TRUE(AmdGpu::IsNumberConversionSupported(AmdGpu::NumberFormat::SnormNz,
                                                    AmdGpu::DataFormat::Format16_16));
    EXPECT_FALSE(AmdGpu::IsNumberConversionSupported(AmdGpu::NumberFormat::SnormNz,
                                                     AmdGpu::DataFormat::Format10_11_11));

    AmdGpu::Buffer buffer{};
    buffer.type = 0;
    buffer.num_format = static_cast<u32>(AmdGpu::NumberFormat::SnormNz);
    buffer.data_format = static_cast<u32>(AmdGpu::DataFormat::Format10_11_11);
    EXPECT_FALSE(buffer.Valid());
    EXPECT_EQ(buffer.GetNumberConversion(), AmdGpu::NumberConversion::None);

    AmdGpu::Image image = AmdGpu::Image::Null(false);
    image.num_format = static_cast<u32>(AmdGpu::NumberFormat::SnormNz);
    image.data_format = static_cast<u32>(AmdGpu::DataFormat::Format10_11_11);
    EXPECT_FALSE(image.Valid());
    EXPECT_EQ(image.GetNumberConversion(), AmdGpu::NumberConversion::None);
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
    EXPECT_NE(helper.find("bool is_storage"), std::string::npos);
    EXPECT_NE(helper.find("image_bindings.emplace_back"), std::string::npos);
    EXPECT_NE(helper.find("desc.type = is_storage ?"), std::string::npos);
    EXPECT_NE(helper.find("BindingType::Storage"), std::string::npos);
    EXPECT_NE(helper.find("BindingType::Texture"), std::string::npos);
    EXPECT_NE(helper.find("image_descriptor_array_sizes.push_back(1)"), std::string::npos);
    EXPECT_NE(helper.find("image_native_extent_requirements.emplace_back(false, false)"),
              std::string::npos);

    size_t call_count = 0;
    size_t call = helper_end;
    while ((call = body.find("append_invalid_image_binding(image_desc.is_written);", call)) !=
           std::string::npos) {
        ++call_count;
        ++call;
    }
    EXPECT_EQ(call_count, 2U);
}

TEST(TextureBindingBookkeeping, RejectsReservedComponentSelectorsBeforeImageCreation) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto function_begin = source.find("void Rasterizer::BindTextures(");
    const auto image_creation = source.find("texture_cache.FindImage(desc)", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(image_creation, std::string::npos);

    const auto prefix = source.substr(function_begin, image_creation - function_begin);
    EXPECT_NE(prefix.find("!tsharp.DstSelect().Valid()"), std::string::npos);
    EXPECT_NE(prefix.find("!tsharp.Valid()"), std::string::npos);
}

TEST(BufferBindingBookkeeping, RejectsInvalidNumberConversionBeforePublication) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto function_begin = source.find("void Rasterizer::BindBuffers(");
    const auto binding_size = source.find("desc.GetBindingSize(vsharp)", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(binding_size, std::string::npos);
    const auto prefix = source.substr(function_begin, binding_size - function_begin);
    EXPECT_NE(prefix.find("vsharp.Valid()"), std::string::npos);
}

TEST(SamplerDescriptor, AcceptsOnlyEncodingsDefinedByTheGuestIsa) {
    AmdGpu::Sampler sampler{};
    sampler.max_lod.Assign(0x100);

    for (u64 ratio = static_cast<u64>(AmdGpu::AnisoRatio::One);
         ratio <= static_cast<u64>(AmdGpu::AnisoRatio::Sixteen); ++ratio) {
        sampler.max_aniso.Assign(static_cast<AmdGpu::AnisoRatio>(ratio));
        EXPECT_TRUE(sampler.Valid()) << ratio;
    }

    for (u64 ratio = 5; ratio < 8; ++ratio) {
        sampler.max_aniso.Assign(static_cast<AmdGpu::AnisoRatio>(ratio));
        EXPECT_FALSE(sampler.Valid()) << ratio;
    }

    sampler.max_aniso.Assign(AmdGpu::AnisoRatio::One);
    sampler.filter_mode.Assign(static_cast<AmdGpu::FilterMode>(3));
    EXPECT_FALSE(sampler.Valid());
    sampler.filter_mode.Assign(AmdGpu::FilterMode::Blend);
    sampler.mip_filter.Assign(static_cast<AmdGpu::MipFilter>(3));
    EXPECT_FALSE(sampler.Valid());
    sampler.mip_filter.Assign(AmdGpu::MipFilter::Linear);
    sampler.min_lod.Assign(0x200);
    EXPECT_FALSE(sampler.Valid());
}

TEST(SamplerDescriptor, InvalidRuntimeDescriptorIsCanonicalizedBeforeVulkanCreation) {
    const auto source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto loop = source.find("for (const auto& sampler : stage.samplers)");
    const auto get_sampler = source.find("texture_cache.GetSampler(ssharp", loop);
    ASSERT_NE(loop, std::string::npos);
    ASSERT_NE(get_sampler, std::string::npos);
    const auto prefix = source.substr(loop, get_sampler - loop);
    EXPECT_NE(prefix.find("if (!ssharp.Valid())"), std::string::npos);
    EXPECT_NE(prefix.find("ssharp = {}"), std::string::npos);
}

TEST(ImageAllocationDiagnostics, ReportsRequestedImageAndHeapBudgetsBeforeAssertion) {
    const auto source = ReadText(SHADPS4_IMAGE_SOURCE_PATH);
    const auto allocation = source.find("vmaCreateImage");
    const auto assertion = source.find("ASSERT_MSG(result", allocation);
    ASSERT_NE(allocation, std::string::npos);
    ASSERT_NE(assertion, std::string::npos);

    const auto failure_path = source.substr(allocation, assertion - allocation);
    EXPECT_NE(failure_path.find("vmaGetHeapBudgets"), std::string::npos);
    EXPECT_NE(failure_path.find("image_ci.extent.width"), std::string::npos);
    EXPECT_NE(failure_path.find("budget.usage"), std::string::npos);
    EXPECT_NE(failure_path.find("budget.budget"), std::string::npos);
}

TEST(TextureCacheDiagnostics, MemoryPressureTelemetryIsOptInAndBounded) {
    const auto source = ReadText(SHADPS4_TEXTURE_CACHE_SOURCE_PATH);
    const auto gc = source.find("void TextureCache::GarbageCollectImages()");
    ASSERT_NE(gc, std::string::npos);
    const auto body = source.substr(gc, source.find("void TextureCache::GarbageCollectSamplers()", gc) - gc);

    EXPECT_NE(body.find("SHADPS4_DIAGNOSTIC_TEXTURE_GC"), std::string::npos);
    EXPECT_NE(body.find("gc_tick % 30"), std::string::npos);
    EXPECT_NE(body.find("slot_images.size()"), std::string::npos);
    EXPECT_NE(body.find("trigger_gc_memory"), std::string::npos);
    EXPECT_NE(body.find("pressure_gc_memory"), std::string::npos);
    EXPECT_NE(body.find("critical_gc_memory"), std::string::npos);
}

TEST(TextureCacheDiagnostics, BurstRegistrationsReportDescriptorAndOverlapIdentity) {
    const auto source = ReadText(SHADPS4_TEXTURE_CACHE_SOURCE_PATH);
    const auto registration = source.find("void TextureCache::RegisterImage(ImageId image_id)");
    const auto unregistration = source.find("void TextureCache::UnregisterImage", registration);
    ASSERT_NE(registration, std::string::npos);
    ASSERT_NE(unregistration, std::string::npos);
    const auto body = source.substr(registration, unregistration - registration);

    EXPECT_NE(body.find("SHADPS4_DIAGNOSTIC_TEXTURE_GC"), std::string::npos);
    EXPECT_NE(body.find("overlap_count"), std::string::npos);
    EXPECT_NE(body.find("exact_address_count"), std::string::npos);
    EXPECT_NE(body.find("image.info.guest_address"), std::string::npos);
    EXPECT_NE(body.find("image.info.guest_size"), std::string::npos);
    EXPECT_NE(body.find("image.info.size.width"), std::string::npos);
    EXPECT_NE(body.find("image.info.resources.levels"), std::string::npos);
    EXPECT_NE(body.find("image.info.array_mode"), std::string::npos);
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
