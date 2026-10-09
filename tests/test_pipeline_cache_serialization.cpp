// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <limits>

#include <gtest/gtest.h>

#include "common/serdes.h"
#include "shader_recompiler/specialization_serialization.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache_serialization.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace {

TEST(PipelineCacheSerialization, SchemaVersionsCoverCurrentWireFormats) {
    EXPECT_EQ(Vulkan::PipelineKeyVersion, 6U);
    EXPECT_EQ(Shader::ShaderMetaVersion, 8U);
    EXPECT_EQ(Shader::ShaderBinaryVersion, 8U);
}

std::vector<u32> MinimalFramedSpirv() {
    return {
        0x07230203U, // Magic
        0x00010600U, // SPIR-V 1.6
        0U,          // Generator
        1U,          // Id bound
        0U,          // Reserved schema
        0x00010000U, // One-word OpNop
    };
}

Vulkan::GraphicsPipelineKey ValidGraphicsKey() {
    Vulkan::GraphicsPipelineKey key{};
    key.stage_hashes[static_cast<u32>(Shader::LogicalStage::Vertex)] = 0x1234;
    key.prim_type = AmdGpu::PrimitiveType::TriangleList;
    key.polygon_mode = AmdGpu::PolygonMode::Fill;
    key.clip_space = AmdGpu::ClipSpace::ZeroToW;
    key.provoking_vtx_last = AmdGpu::ProvokingVtxLast::First;
    key.logic_op = AmdGpu::ColorControl::LogicOp::Copy;
    key.num_samples = 1;
    key.depth_samples = 1;
    return key;
}

Vulkan::GraphicsPipeline::SerializationSupport ValidSerializationSupport() {
    Vulkan::GraphicsPipeline::SerializationSupport support{};
    support.multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    return support;
}

TEST(PipelineCacheSerialization, GraphicsRecordRoundTripsAuxiliaryFragment) {
    const auto expected_key = ValidGraphicsKey();
    auto expected_support = ValidSerializationSupport();
    expected_support.fragment = MinimalFramedSpirv();

    Serialization::Archive encoded;
    expected_key.Serialize(encoded);
    expected_support.Serialize(encoded);
    Serialization::Archive input{encoded.TakeOff()};
    Vulkan::GraphicsPipelineKey actual_key{};
    Vulkan::GraphicsPipeline::SerializationSupport actual_support{};
    ASSERT_TRUE(Vulkan::DeserializeGraphicsPipelineCacheRecord(input, actual_key, actual_support));
    EXPECT_EQ(actual_key, expected_key);
    EXPECT_EQ(actual_support.fragment, expected_support.fragment);
    EXPECT_TRUE(input.IsEoS());
}

TEST(PipelineCacheSerialization, TrailingDataDoesNotPublishPartialGraphicsRecord) {
    const auto expected_key = ValidGraphicsKey();
    auto expected_support = ValidSerializationSupport();
    expected_support.fragment = MinimalFramedSpirv();

    Serialization::Archive encoded;
    expected_key.Serialize(encoded);
    expected_support.Serialize(encoded);
    Serialization::Writer writer{encoded};
    writer.Write(u8{0xA5});

    auto actual_key = ValidGraphicsKey();
    actual_key.stage_hashes[static_cast<u32>(Shader::LogicalStage::Vertex)] = 0xDEADBEEF;
    auto actual_support = ValidSerializationSupport();
    actual_support.fragment = {0xBADU};
    const auto unchanged_key = actual_key;
    const auto unchanged_fragment = actual_support.fragment;

    Serialization::Archive input{encoded.TakeOff()};
    EXPECT_FALSE(Vulkan::DeserializeGraphicsPipelineCacheRecord(input, actual_key, actual_support));
    EXPECT_EQ(actual_key, unchanged_key);
    EXPECT_EQ(actual_support.fragment, unchanged_fragment);
}

TEST(PipelineCacheSerialization, OversizedAuxiliaryFragmentFailsWithoutPublication) {
    const auto expected_key = ValidGraphicsKey();
    auto oversized = ValidSerializationSupport();
    oversized.fragment.resize(Vulkan::MaxCachedShaderDwords + 1, 0U);

    Serialization::Archive encoded;
    expected_key.Serialize(encoded);
    oversized.Serialize(encoded);
    Serialization::Archive input{encoded.TakeOff()};
    Vulkan::GraphicsPipelineKey actual_key{};
    auto actual_support = ValidSerializationSupport();
    actual_support.fragment = {0xBADU};
    EXPECT_FALSE(Vulkan::DeserializeGraphicsPipelineCacheRecord(input, actual_key, actual_support));
    EXPECT_EQ(actual_support.fragment, std::vector<u32>{0xBADU});
}

TEST(PipelineCacheSerialization, GraphicsKeyRejectsInvalidDomainsCountsAndMasks) {
    const auto valid = ValidGraphicsKey();
    ASSERT_TRUE(Vulkan::ValidateGraphicsPipelineKey(valid));

    const auto expect_rejected = [&](const char* field, auto mutate) {
        SCOPED_TRACE(field);
        auto invalid = valid;
        mutate(invalid);
        EXPECT_FALSE(Vulkan::ValidateGraphicsPipelineKey(invalid));
    };
    expect_rejected("compute stage", [](auto& key) {
        key.stage_hashes[static_cast<u32>(Shader::LogicalStage::Compute)] = 1;
    });
    expect_rejected("missing vertex stage", [](auto& key) {
        key.stage_hashes[static_cast<u32>(Shader::LogicalStage::Vertex)] = 0;
    });
    expect_rejected("unpaired tessellation", [](auto& key) {
        key.stage_hashes[static_cast<u32>(Shader::LogicalStage::TessellationControl)] = 1;
    });
    expect_rejected("primitive domain",
                    [](auto& key) { key.prim_type = static_cast<AmdGpu::PrimitiveType>(31); });
    for (const auto unsupported : {AmdGpu::PrimitiveType::None, AmdGpu::PrimitiveType::LineLoop,
                                   AmdGpu::PrimitiveType::QuadStrip}) {
        SCOPED_TRACE(static_cast<u32>(unsupported));
        auto invalid = valid;
        invalid.prim_type = unsupported;
        EXPECT_FALSE(Vulkan::ValidateGraphicsPipelineKey(invalid));
    }
    expect_rejected("polygon domain",
                    [](auto& key) { key.polygon_mode = static_cast<AmdGpu::PolygonMode>(3); });
    expect_rejected("depth format domain",
                    [](auto& key) { key.z_format = static_cast<AmdGpu::DepthBuffer::ZFormat>(2); });
    expect_rejected("attachment count",
                    [](auto& key) { key.num_color_attachments = AmdGpu::NUM_COLOR_BUFFERS + 1; });
    expect_rejected("mrt mask", [](auto& key) { key.mrt_mask = 1U << AmdGpu::NUM_COLOR_BUFFERS; });
    expect_rejected("count and mask disagreement", [](auto& key) {
        key.num_color_attachments = 1;
        key.mrt_mask = 0;
    });
    expect_rejected("write mask",
                    [](auto& key) { key.write_masks[0] = vk::ColorComponentFlags{0x10}; });
    expect_rejected("sample count", [](auto& key) { key.num_samples = 3; });
    expect_rejected("logic operation", [](auto& key) {
        key.logic_op = static_cast<AmdGpu::ColorControl::LogicOp>(1);
    });

    auto invalid_support = ValidSerializationSupport();
    invalid_support.multisampling.rasterizationSamples = vk::SampleCountFlagBits::e64;
    Serialization::Archive encoded;
    valid.Serialize(encoded);
    invalid_support.Serialize(encoded);
    Serialization::Archive input{encoded.TakeOff()};
    Vulkan::GraphicsPipelineKey decoded_key{};
    Vulkan::GraphicsPipeline::SerializationSupport decoded_support{};
    EXPECT_FALSE(
        Vulkan::DeserializeGraphicsPipelineCacheRecord(input, decoded_key, decoded_support));
}

TEST(PipelineCacheSerialization, GraphicsKeyRejectsNoncanonicalReservedBits) {
    const auto valid = ValidGraphicsKey();
    Serialization::Archive encoded;
    valid.Serialize(encoded);
    auto bytes = encoded.TakeOff();
    ASSERT_EQ(bytes.size(), sizeof(Vulkan::GraphicsPipelineKey));
    bytes.back() ^= 0x80U;

    auto decoded = valid;
    decoded.stage_hashes[static_cast<u32>(Shader::LogicalStage::Vertex)] = 0xDEADBEEF;
    const auto unchanged = decoded;
    Serialization::Archive input{std::move(bytes)};
    EXPECT_FALSE(decoded.Deserialize(input));
    EXPECT_EQ(decoded, unchanged);
}

TEST(PipelineCacheSerialization, SpirvFramingRejectsMalformedAndOversizedInput) {
    const auto valid = MinimalFramedSpirv();
    ASSERT_TRUE(Vulkan::IsValidCachedSpirv(valid));

    auto bad_magic = valid;
    bad_magic[0] = 0;
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(bad_magic));
    auto bad_schema = valid;
    bad_schema[4] = 1;
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(bad_schema));
    auto zero_word_count = valid;
    zero_word_count[5] = 0;
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(zero_word_count));
    auto truncated_instruction = valid;
    truncated_instruction[5] = 0x00020000U;
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(truncated_instruction));
    std::vector<u32> oversized(Vulkan::MaxCachedShaderDwords + 1, 0U);
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(oversized));
}

TEST(PipelineCacheSerialization, BlobCapsApplyBeforeTypedAllocation) {
    for (const auto type : {Storage::BlobType::ShaderMeta, Storage::BlobType::ShaderBinary,
                            Storage::BlobType::PipelineKey, Storage::BlobType::ShaderProfile}) {
        SCOPED_TRACE(static_cast<u32>(type));
        const size_t cap = Storage::MaxCacheBlobBytes(type);
        EXPECT_TRUE(Storage::IsCacheBlobSizeValid(type, cap, 1));
        EXPECT_FALSE(Storage::IsCacheBlobSizeValid(type, cap + 1, 1));
    }
    EXPECT_TRUE(Storage::IsCacheBlobSizeValid(Storage::BlobType::ShaderBinary, 16, sizeof(u32)));
    EXPECT_FALSE(Storage::IsCacheBlobSizeValid(Storage::BlobType::ShaderBinary, 15, sizeof(u32)));
    EXPECT_FALSE(Storage::IsCacheBlobSizeValid(Storage::BlobType::ShaderBinary, 16, 0));
    EXPECT_FALSE(Storage::IsCacheBlobSizeValid(Storage::BlobType::PipelineKey,
                                               std::numeric_limits<u64>::max(), 1));

    std::vector<u32> target{0xBADU};
    const auto original_capacity = target.capacity();
    EXPECT_FALSE(Storage::PrepareCacheBlobBuffer(
        Storage::BlobType::ShaderBinary,
        Storage::MaxCacheBlobBytes(Storage::BlobType::ShaderBinary) + sizeof(u32), target));
    EXPECT_TRUE(target.empty());
    EXPECT_EQ(target.capacity(), original_capacity);
    EXPECT_TRUE(Storage::PrepareCacheBlobBuffer(Storage::BlobType::ShaderBinary, 16, target));
    EXPECT_EQ(target.size(), 4U);
}

} // namespace
