// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "video_core/texture_cache/types.h"

namespace {

using VideoCore::SubresourceExtent;

std::string ReadText(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
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
    EXPECT_NE(helper.find("image_bindings.emplace_back"), std::string::npos);
    EXPECT_NE(helper.find("image_descriptor_array_sizes.push_back(1)"), std::string::npos);
    EXPECT_NE(helper.find("image_native_extent_requirements.emplace_back(false, false)"),
              std::string::npos);
    EXPECT_NE(helper.find("BindingType::Storage"), std::string::npos);

    size_t call_count = 0;
    size_t call = helper_end;
    while ((call = body.find("append_invalid_image_binding(image_desc.is_written);", call)) !=
           std::string::npos) {
        ++call_count;
        ++call;
    }
    EXPECT_EQ(call_count, 2U);
}

TEST(SpirvFragCoordScaling, ReadsHostScaleThroughAPushConstantPointer) {
    const auto source = ReadText(SHADPS4_SPIRV_CONTEXT_GET_SET_SOURCE_PATH);
    const auto frag_coord = source.find("case IR::Attribute::FragCoord:");
    const auto next_case = source.find("case IR::Attribute::TessellationEvaluationPointU:",
                                       frag_coord);
    ASSERT_NE(frag_coord, std::string::npos);
    ASSERT_NE(next_case, std::string::npos);
    const auto body = source.substr(frag_coord, next_case - frag_coord);

    EXPECT_NE(body.find("TypePointer(spv::StorageClass::PushConstant, ctx.F32[1])"),
              std::string::npos);
    EXPECT_EQ(body.find("OpAccessChain(ctx.input_f32, ctx.push_data_block"), std::string::npos);
}

TEST(VulkanFeatureNegotiation, EnablesSupportedIndirectFirstInstance) {
    const auto source = ReadText(SHADPS4_VK_INSTANCE_SOURCE_PATH);
    EXPECT_NE(source.find(".drawIndirectFirstInstance = features.drawIndirectFirstInstance"),
              std::string::npos);
}

} // namespace
