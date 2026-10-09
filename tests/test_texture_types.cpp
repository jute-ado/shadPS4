// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "video_core/amdgpu/resource.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/resource.h"
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

TEST(InvalidImageAccesses, TracksInvalidResourcesWithoutDisablingOtherSideEffects) {
    Shader::PushData push{};
    push.ClearInvalidImages();
    EXPECT_FALSE(push.IsImageInvalid(1));
    EXPECT_FALSE(push.IsImageInvalid(7));
    EXPECT_FALSE(push.IsImageInvalid(47));

    push.MarkInvalidImage(1);
    push.MarkInvalidImage(7);
    push.MarkInvalidImage(47);

    EXPECT_TRUE(push.IsImageInvalid(1));
    EXPECT_TRUE(push.IsImageInvalid(7));
    EXPECT_TRUE(push.IsImageInvalid(47));
    EXPECT_FALSE(push.IsImageInvalid(0));
}

TEST(InvalidImageAccesses, BindingAndSpirvEmissionShareTheValidityMask) {
    const auto rasterizer = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto image_emitter = ReadText(SHADPS4_SPIRV_IMAGE_SOURCE_PATH);
    const auto sample_begin = image_emitter.find("Id EmitImageSampleExplicitLod(");
    const auto read_begin = image_emitter.find("Id EmitImageRead(");
    const auto write_begin = image_emitter.find("void EmitImageWrite(");
    ASSERT_NE(sample_begin, std::string::npos);
    ASSERT_NE(read_begin, std::string::npos);
    ASSERT_NE(write_begin, std::string::npos);
    const auto sample_body = image_emitter.substr(sample_begin, read_begin - sample_begin);
    const auto read_body = image_emitter.substr(read_begin, write_begin - read_begin);
    const auto write_body = image_emitter.substr(write_begin);

    EXPECT_NE(rasterizer.find("MarkInvalidImage"), std::string::npos);
    EXPECT_NE(image_emitter.find("PushData::InvalidImagesIndex"), std::string::npos);
    for (const auto* body : {&sample_body, &read_body}) {
        const auto validity_guard = body->find("EmitComputeImageAccessIfValid");
        const auto image_descriptor_load = body->find("OpLoad(texture.image_type");
        ASSERT_NE(validity_guard, std::string::npos);
        ASSERT_NE(image_descriptor_load, std::string::npos);
        EXPECT_LT(validity_guard, image_descriptor_load);
    }
    const auto validity_branch = write_body.find("OpBranchConditional");
    const auto image_descriptor_load = write_body.find("OpLoad(texture.image_type");
    ASSERT_NE(validity_branch, std::string::npos);
    ASSERT_NE(image_descriptor_load, std::string::npos);
    EXPECT_LT(validity_branch, image_descriptor_load);
    EXPECT_NE(write_body.find("OpImageWrite"), std::string::npos);
}

} // namespace
