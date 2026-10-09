// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "shader_recompiler/backend/bindings.h"
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
    EXPECT_NE(helper.find("image_descriptor_array_sizes.push_back(1)"), std::string::npos);
    EXPECT_NE(helper.find("image_native_extent_requirements.emplace_back(false, false)"),
              std::string::npos);

    size_t call_count = 0;
    size_t call = helper_end;
    while ((call = body.find("append_invalid_image_binding();", call)) != std::string::npos) {
        ++call_count;
        ++call;
    }
    EXPECT_EQ(call_count, 2U);
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
