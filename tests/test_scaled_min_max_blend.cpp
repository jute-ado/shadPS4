// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <string>

#include "video_core/amdgpu/regs_color.h"
#include "video_core/renderer_vulkan/scaled_min_max_blend.h"

namespace {

using AmdGpu::BlendControl;

std::string ReadSource(const char* path) {
    std::ifstream source{path, std::ios::binary};
    EXPECT_TRUE(source.is_open());
    return {std::istreambuf_iterator<char>{source}, std::istreambuf_iterator<char>{}};
}

BlendControl SelfScaled(BlendControl::BlendFunc function) {
    BlendControl control{};
    control.enable = 1;
    control.color_func = function;
    control.color_src_factor = BlendControl::BlendFactor::SrcColor;
    control.color_dst_factor = BlendControl::BlendFactor::DstColor;
    return control;
}

TEST(ScaledMinMaxBlend, RecognizesOnlyEnabledSelfScaledColorOperation) {
    EXPECT_TRUE(Vulkan::IsSelfScaledMinMaxBlend(SelfScaled(BlendControl::BlendFunc::Min), false));
    EXPECT_TRUE(Vulkan::IsSelfScaledMinMaxBlend(SelfScaled(BlendControl::BlendFunc::Max), false));

    auto disabled = SelfScaled(BlendControl::BlendFunc::Min);
    disabled.enable = 0;
    EXPECT_FALSE(Vulkan::IsSelfScaledMinMaxBlend(disabled, false));
    EXPECT_FALSE(Vulkan::IsSelfScaledMinMaxBlend(SelfScaled(BlendControl::BlendFunc::Add), false));
    EXPECT_FALSE(Vulkan::IsSelfScaledMinMaxBlend(SelfScaled(BlendControl::BlendFunc::Min), true));
}

TEST(ScaledMinMaxBlend, RejectsNonSelfScaledFactors) {
    auto control = SelfScaled(BlendControl::BlendFunc::Max);
    control.color_src_factor = BlendControl::BlendFactor::One;
    EXPECT_FALSE(Vulkan::IsSelfScaledMinMaxBlend(control, false));

    control = SelfScaled(BlendControl::BlendFunc::Max);
    control.color_dst_factor = BlendControl::BlendFactor::One;
    EXPECT_FALSE(Vulkan::IsSelfScaledMinMaxBlend(control, false));
}

// These source-contract checks intentionally sit beside the policy tests: a correct predicate is
// not enough unless the pipeline key, shader export, and Vulkan attachment all consume it.
TEST(ScaledMinMaxBlend, PipelineCapturesAndConsumesSelfScaleContract) {
    const auto cache = ReadSource(SHADPS4_SCALED_MIN_MAX_PIPELINE_CACHE_SOURCE_PATH);
    const auto graphics = ReadSource(SHADPS4_SCALED_MIN_MAX_GRAPHICS_PIPELINE_SOURCE_PATH);
    const auto spirv = ReadSource(SHADPS4_SCALED_MIN_MAX_SPIRV_SOURCE_PATH);

    EXPECT_NE(cache.find("IsSelfScaledMinMaxBlend"), std::string::npos);
    EXPECT_NE(cache.find("color_buffer.blend_self_scale"), std::string::npos);
    EXPECT_NE(graphics.find("!key.color_buffers[i].blend_self_scale"), std::string::npos);
    EXPECT_NE(graphics.find("attachments[i].srcColorBlendFactor = vk::BlendFactor::eOne"),
              std::string::npos);
    EXPECT_NE(graphics.find("attachments[i].dstColorBlendFactor = vk::BlendFactor::eOne"),
              std::string::npos);
    EXPECT_NE(spirv.find("element < 3 && ctx.runtime_info.fs_info.color_buffers[index].blend_self_scale"),
              std::string::npos);
    EXPECT_NE(spirv.find("ctx.OpFMul(ctx.F32[1], value, value)"), std::string::npos);
}

TEST(ScaledMinMaxBlend, RuntimeAndPipelineSerializationPreserveSelfScaleBit) {
    const auto runtime = ReadSource(SHADPS4_SCALED_MIN_MAX_RUNTIME_SERIALIZATION_SOURCE_PATH);
    const auto identity = ReadSource(SHADPS4_SCALED_MIN_MAX_IDENTITY_SOURCE_PATH);
    const auto pipeline = ReadSource(SHADPS4_SCALED_MIN_MAX_PIPELINE_SERIALIZATION_SOURCE_PATH);

    EXPECT_NE(runtime.find("WriteBool(writer, color.blend_self_scale)"), std::string::npos);
    EXPECT_NE(runtime.find("ReadBool(reader, blend_self_scale)"), std::string::npos);
    EXPECT_NE(identity.find("writer.WriteBool(color.blend_self_scale)"), std::string::npos);
    EXPECT_NE(pipeline.find("canonical_color.blend_self_scale = color.blend_self_scale"),
              std::string::npos);
}

} // namespace
