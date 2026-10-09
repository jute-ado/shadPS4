// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "common/serdes.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"

namespace Vulkan {
namespace {

TEST(GraphicsPipelineSerialization, RoundTripsGeneratedAuxiliaryShaderBytecode) {
    GraphicsPipeline::SerializationSupport original{};
    original.vertex_attributes.emplace_back(vk::VertexInputAttributeDescription{
        .location = 2,
        .binding = 3,
        .format = vk::Format::eR32G32Sfloat,
        .offset = 16,
    });
    original.vertex_bindings.emplace_back(vk::VertexInputBindingDescription{
        .binding = 3,
        .stride = 24,
        .inputRate = vk::VertexInputRate::eInstance,
    });
    original.divisors.emplace_back(vk::VertexInputBindingDivisorDescriptionEXT{
        .binding = 3,
        .divisor = 7,
    });
    original.multisampling.rasterizationSamples = vk::SampleCountFlagBits::e4;
    original.multisampling.sampleShadingEnable = true;
    original.multisampling.minSampleShading = 0.5F;
    original.multisampling.alphaToCoverageEnable = true;
    original.tcs = {0x07230203U, 11U, 12U};
    original.tes = {0x07230203U, 21U, 22U};
    original.fragment = {0x07230203U, 31U, 32U, 33U};

    Serialization::Archive output;
    original.Serialize(output);
    Serialization::Archive input{output.TakeOff()};
    GraphicsPipeline::SerializationSupport restored{};
    ASSERT_TRUE(restored.Deserialize(input));

    ASSERT_EQ(restored.vertex_attributes.size(), 1U);
    EXPECT_EQ(restored.vertex_attributes[0].location, 2U);
    EXPECT_EQ(restored.vertex_attributes[0].binding, 3U);
    EXPECT_EQ(restored.vertex_attributes[0].format, vk::Format::eR32G32Sfloat);
    EXPECT_EQ(restored.vertex_attributes[0].offset, 16U);
    ASSERT_EQ(restored.vertex_bindings.size(), 1U);
    EXPECT_EQ(restored.vertex_bindings[0].stride, 24U);
    EXPECT_EQ(restored.vertex_bindings[0].inputRate, vk::VertexInputRate::eInstance);
    ASSERT_EQ(restored.divisors.size(), 1U);
    EXPECT_EQ(restored.divisors[0].divisor, 7U);
    EXPECT_EQ(restored.multisampling.rasterizationSamples, vk::SampleCountFlagBits::e4);
    EXPECT_TRUE(restored.multisampling.sampleShadingEnable);
    EXPECT_FLOAT_EQ(restored.multisampling.minSampleShading, 0.5F);
    EXPECT_TRUE(restored.multisampling.alphaToCoverageEnable);
    EXPECT_EQ(restored.tcs, original.tcs);
    EXPECT_EQ(restored.tes, original.tes);
    EXPECT_EQ(restored.fragment, original.fragment);
}

} // namespace
} // namespace Vulkan
