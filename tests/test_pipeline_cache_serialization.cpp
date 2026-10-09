// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <limits>
#include <optional>

#include <gtest/gtest.h>

#include "common/serdes.h"
#include "shader_recompiler/backend/spirv/emit_spirv_discard_frag.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/specialization_serialization.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache_serialization.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace {

TEST(PipelineCacheSerialization, SchemaVersionsCoverCurrentWireFormats) {
    EXPECT_EQ(Vulkan::PipelineKeyVersion, 6U);
    EXPECT_EQ(Shader::ShaderMetaVersion, 7U);
    EXPECT_EQ(Shader::ShaderBinaryVersion, 5U);
}

std::vector<u32> MinimalValidSpirv() {
    return {
        0x07230203U,                           // Magic
        0x00010000U,                           // SPIR-V 1.0
        0U,                                    // Generator
        6U,                                    // Id bound
        0U,                                    // Reserved schema
        0x00020011U, 0x00000001U,              // OpCapability Shader
        0x0003000EU, 0x00000000U, 0x00000001U, // OpMemoryModel Logical GLSL450
        0x0005000FU, 0x00000005U, 0x00000004U, 0x6E69616DU,
        0x00000000U, // OpEntryPoint GLCompute %4 "main"
        0x00060010U, 0x00000004U, 0x00000011U, 0x00000001U, 0x00000001U,
        0x00000001U,                           // OpExecutionMode %4 LocalSize 1 1 1
        0x00020013U, 0x00000001U,              // %1 = OpTypeVoid
        0x00030021U, 0x00000002U, 0x00000001U, // %2 = OpTypeFunction %1
        0x00050036U, 0x00000001U, 0x00000004U, 0x00000000U,
        0x00000002U,              // %4 = OpFunction %1 None %2
        0x000200F8U, 0x00000005U, // %5 = OpLabel
        0x000100FDU,              // OpReturn
        0x00010038U,              // OpFunctionEnd
    };
}

std::optional<size_t> FindSpirvInstruction(std::span<const u32> code, u16 opcode) {
    for (size_t offset = 5; offset < code.size();) {
        const size_t word_count = code[offset] >> 16;
        if (word_count == 0 || word_count > code.size() - offset) {
            return std::nullopt;
        }
        if ((code[offset] & 0xFFFFU) == opcode) {
            return offset;
        }
        offset += word_count;
    }
    return std::nullopt;
}

bool EraseSpirvInstruction(std::vector<u32>& code, u16 opcode) {
    const auto offset = FindSpirvInstruction(code, opcode);
    if (!offset) {
        return false;
    }
    const size_t word_count = code[*offset] >> 16;
    code.erase(code.begin() + *offset, code.begin() + *offset + word_count);
    return true;
}

Vulkan::GraphicsPipelineKey ValidGraphicsKey() {
    Vulkan::GraphicsPipelineKey key{};
    key.stage_hashes[static_cast<u32>(Shader::LogicalStage::Vertex)] = 0x1234;
    key.stage_hashes[static_cast<u32>(Shader::LogicalStage::Fragment)] = 0x5678;
    key.vertex_buffer_formats[0] = vk::Format::eR32G32Sfloat;
    key.num_color_attachments = 1;
    key.color_buffers[0].data_format = AmdGpu::DataFormat::Format8_8_8_8;
    key.color_buffers[0].num_format = AmdGpu::NumberFormat::Unorm;
    key.color_buffers[0].num_conversion = AmdGpu::NumberConversion::None;
    key.color_buffers[0].export_format = AmdGpu::ShaderExportFormat::ABGR_32;
    key.color_buffers[0].swizzle.r = AmdGpu::CompSwizzle::Blue;
    key.color_buffers[0].swizzle.g = AmdGpu::CompSwizzle::Green;
    key.color_buffers[0].swizzle.b = AmdGpu::CompSwizzle::Red;
    key.color_buffers[0].swizzle.a = AmdGpu::CompSwizzle::One;
    key.blend_controls[0].color_src_factor = AmdGpu::BlendControl::BlendFactor::SrcAlpha;
    key.blend_controls[0].color_func = AmdGpu::BlendControl::BlendFunc::ReverseSubtract;
    key.blend_controls[0].color_dst_factor = AmdGpu::BlendControl::BlendFactor::OneMinusSrcAlpha;
    key.blend_controls[0].alpha_src_factor = AmdGpu::BlendControl::BlendFactor::One;
    key.blend_controls[0].alpha_func = AmdGpu::BlendControl::BlendFunc::Max;
    key.blend_controls[0].alpha_dst_factor = AmdGpu::BlendControl::BlendFactor::Zero;
    key.blend_controls[0].separate_alpha_blend = 1;
    key.blend_controls[0].enable = 1;
    key.blend_controls[0].disable_rop3 = 1;
    key.write_masks[0] = vk::ColorComponentFlags{0xFU};
    key.cb_shader_mask.raw = 0xFU;
    key.logic_op = AmdGpu::ColorControl::LogicOp::Xor;
    key.num_samples = 1;
    key.depth_samples = 1;
    key.color_samples[0] = 1;
    key.mrt_mask = 1;
    key.z_format = AmdGpu::DepthBuffer::ZFormat::Z32Float;
    key.stencil_format = AmdGpu::DepthBuffer::StencilFormat::Stencil8;
    key.depth_clamp_enable = 1;
    key.prim_type = AmdGpu::PrimitiveType::TriangleStrip;
    key.polygon_mode = AmdGpu::PolygonMode::Line;
    key.clip_space = AmdGpu::ClipSpace::MinusWToW;
    key.provoking_vtx_last = AmdGpu::ProvokingVtxLast::Last;
    key.depth_clip_enable = 1;
    return key;
}

Vulkan::GraphicsPipeline::SerializationSupport ValidSerializationSupport() {
    Vulkan::GraphicsPipeline::SerializationSupport support{};
    support.vertex_attributes.push_back({
        .location = 0,
        .binding = 0,
        .format = vk::Format::eR32G32Sfloat,
        .offset = 0,
    });
    support.vertex_bindings.push_back({
        .binding = 0,
        .stride = 8,
        .inputRate = vk::VertexInputRate::eVertex,
    });
    support.multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    return support;
}

Vulkan::GraphicsPipeline::SerializationSupport MinimalSerializationSupport() {
    Vulkan::GraphicsPipeline::SerializationSupport support{};
    support.multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    return support;
}

TEST(PipelineCacheSerialization, GraphicsRecordRoundTripsAuxiliaryFragment) {
    const auto expected_key = ValidGraphicsKey();
    auto expected_support = ValidSerializationSupport();
    expected_support.fragment = MinimalValidSpirv();

    Serialization::Archive encoded;
    expected_key.Serialize(encoded);
    expected_support.Serialize(encoded);
    Serialization::Archive input{encoded.TakeOff()};
    Vulkan::GraphicsPipelineKey actual_key{};
    Vulkan::GraphicsPipeline::SerializationSupport actual_support{};
    ASSERT_TRUE(Vulkan::DeserializeGraphicsPipelineCacheRecord(input, actual_key, actual_support));
    EXPECT_EQ(actual_key, expected_key);
    EXPECT_EQ(actual_support.fragment, expected_support.fragment);
    ASSERT_EQ(actual_support.vertex_attributes.size(), 1U);
    EXPECT_EQ(actual_support.vertex_attributes[0].format, vk::Format::eR32G32Sfloat);
    ASSERT_EQ(actual_support.vertex_bindings.size(), 1U);
    EXPECT_EQ(actual_support.vertex_bindings[0].stride, 8U);
    EXPECT_TRUE(input.IsEoS());
}

TEST(PipelineCacheSerialization, TrailingDataDoesNotPublishPartialGraphicsRecord) {
    const auto expected_key = ValidGraphicsKey();
    auto expected_support = ValidSerializationSupport();
    expected_support.fragment = MinimalValidSpirv();

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

TEST(PipelineCacheSerialization, GraphicsKeyUsesFixedWidthFieldsAndCanonicalBooleans) {
    const auto valid = ValidGraphicsKey();
    Serialization::Archive encoded;
    valid.Serialize(encoded);
    auto bytes = encoded.TakeOff();
    // Six u64 hashes, 32 u32 vertex formats, explicit color/blend records and explicit
    // u32/u8 tail fields. Native size_t, bitfield layout and padding are not part of the wire.
    ASSERT_EQ(bytes.size(), 736U);

    constexpr size_t DepthClampEnableOffset = 718;
    ASSERT_LT(DepthClampEnableOffset, bytes.size());
    bytes[DepthClampEnableOffset] = 2U;

    auto decoded = valid;
    decoded.stage_hashes[static_cast<u32>(Shader::LogicalStage::Vertex)] = 0xDEADBEEF;
    const auto unchanged = decoded;
    Serialization::Archive input{std::move(bytes)};
    EXPECT_FALSE(decoded.Deserialize(input));
    EXPECT_EQ(decoded, unchanged);

    Serialization::Archive encoded_again;
    valid.Serialize(encoded_again);
    auto invalid_enum = encoded_again.TakeOff();
    constexpr size_t PrimitiveTypeOffset = 719;
    const u32 out_of_range_primitive = std::numeric_limits<u32>::max();
    std::memcpy(invalid_enum.data() + PrimitiveTypeOffset, &out_of_range_primitive,
                sizeof(out_of_range_primitive));
    Serialization::Archive invalid_enum_input{std::move(invalid_enum)};
    EXPECT_FALSE(decoded.Deserialize(invalid_enum_input));
    EXPECT_EQ(decoded, unchanged);
}

TEST(PipelineCacheSerialization, SupportUsesU32CountsInsteadOfNativeSizeT) {
    const auto support = MinimalSerializationSupport();
    Serialization::Archive encoded;
    support.Serialize(encoded);
    // Six u32 vector counts plus six explicit u32 multisample fields.
    EXPECT_EQ(encoded.SizeBytes(), 48U);
}

TEST(PipelineCacheSerialization, ComputeKeyUsesFixedWidthU64) {
    Vulkan::ComputePipelineKey expected{.value = static_cast<size_t>(0x123456789ABCDEF0ULL)};
    Serialization::Archive encoded;
    expected.Serialize(encoded);
    EXPECT_EQ(encoded.SizeBytes(), sizeof(u64));

    Serialization::Archive input{encoded.TakeOff()};
    Vulkan::ComputePipelineKey actual{};
    ASSERT_TRUE(actual.Deserialize(input));
    EXPECT_EQ(actual, expected);
    EXPECT_TRUE(input.IsEoS());
}

TEST(PipelineCacheSerialization, ShaderProfileUsesCanonicalFixedWidthFields) {
    Shader::Profile expected{};
    expected.max_ubo_size = 0x123456789ABCDEF0ULL;
    expected.max_viewport_width = 1920;
    expected.max_viewport_height = 1080;
    expected.max_shared_memory_size = 32768;
    expected.supported_spirv = 0x00010600;
    expected.subgroup_size = 64;
    expected.support_int8 = true;
    expected.support_int16 = true;
    expected.support_int64 = true;
    expected.support_float16 = true;
    expected.support_float64 = true;
    expected.supports_denorm_behavior_independence = true;
    expected.supports_rounding_mode_independence = true;
    expected.support_fp16_denorm_preserve = true;
    expected.support_fp16_denorm_flush = true;
    expected.support_fp16_round_to_zero = true;
    expected.support_fp32_denorm_preserve = true;
    expected.support_fp32_denorm_flush = true;
    expected.support_fp32_round_to_zero = true;
    expected.support_fp64_denorm_preserve = true;
    expected.support_fp64_denorm_flush = true;
    expected.support_fp64_round_to_zero = true;
    expected.support_fp16_signed_zero_inf_nan_preserve = true;
    expected.support_fp32_signed_zero_inf_nan_preserve = true;
    expected.support_fp64_signed_zero_inf_nan_preserve = true;
    expected.supports_image_load_store_lod = true;
    expected.supports_native_cube_calc = true;
    expected.supports_trinary_minmax = true;
    expected.supports_buffer_fp32_atomic_min_max = true;
    expected.supports_image_fp32_atomic_min_max = true;
    expected.supports_buffer_int64_atomics = true;
    expected.supports_shared_int64_atomics = true;
    expected.supports_workgroup_explicit_memory_layout = true;
    expected.supports_amd_shader_explicit_vertex_parameter = true;
    expected.supports_fragment_shader_barycentric = true;
    expected.has_broken_spirv_clamp = true;
    expected.lower_left_origin_mode = true;
    expected.needs_manual_interpolation = true;
    expected.needs_lds_barriers = true;
    expected.needs_buffer_offsets = true;
    expected.needs_unorm_fixup = true;
    expected.needs_clip_distance_emulation = true;
    expected.supports_shader_stencil_export = true;

    Serialization::Archive encoded;
    Vulkan::SerializeShaderProfile(encoded, expected);
    auto bytes = encoded.TakeOff();
    // Version + u64 + five u32 values + 37 canonical u8 booleans.
    ASSERT_EQ(bytes.size(), 69U);
    constexpr size_t FirstBooleanOffset = sizeof(u32) + sizeof(u64) + 5 * sizeof(u32);
    EXPECT_EQ(bytes[FirstBooleanOffset], 1U);

    Serialization::Archive input{std::vector<u8>{bytes}};
    Shader::Profile actual{};
    ASSERT_TRUE(Vulkan::DeserializeShaderProfile(input, actual));
    EXPECT_EQ(actual, expected);

    bytes[FirstBooleanOffset] = 2U;
    Shader::Profile unchanged{};
    unchanged.max_ubo_size = 7;
    const auto original = unchanged;
    Serialization::Archive malformed{std::move(bytes)};
    EXPECT_FALSE(Vulkan::DeserializeShaderProfile(malformed, unchanged));
    EXPECT_EQ(unchanged, original);
}

TEST(PipelineCacheSerialization, SpirvSemanticValidationRejectsIncompleteModule) {
    const auto valid = MinimalValidSpirv();
    ASSERT_TRUE(Vulkan::IsValidCachedSpirv(valid));

    const std::vector<u32> framed_but_incomplete{
        0x07230203U, 0x00010600U, 0U, 1U, 0U, 0x00010000U,
    };
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(framed_but_incomplete));

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
    truncated_instruction.back() = 0x00020038U;
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(truncated_instruction));
    auto malformed_known_instruction = valid;
    const auto return_instruction = FindSpirvInstruction(malformed_known_instruction, 253);
    ASSERT_TRUE(return_instruction);
    malformed_known_instruction[*return_instruction] = 0x0001003EU;
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(malformed_known_instruction));
    std::vector<u32> oversized(Vulkan::MaxCachedShaderDwords + 1, 0U);
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(oversized));
}

TEST(PipelineCacheSerialization, SpirvPreflightAcceptsCurrentSiritDialect) {
    std::array<Shader::OutputMap, 3> outputs{};
    EXPECT_TRUE(
        Vulkan::IsValidCachedSpirv(Shader::Backend::SPIRV::EmitDiscardFragmentShader(outputs)));

    outputs[0][0] = Shader::Output::ClipDist0;
    EXPECT_TRUE(
        Vulkan::IsValidCachedSpirv(Shader::Backend::SPIRV::EmitDiscardFragmentShader(outputs)));
}

TEST(PipelineCacheSerialization, SpirvPreflightRejectsUnsafeModuleSemantics) {
    constexpr u16 OpCapability = 17;
    constexpr u16 OpMemoryModel = 14;
    constexpr u16 OpEntryPoint = 15;
    constexpr u16 OpFunction = 54;
    constexpr u16 OpLabel = 248;
    constexpr u16 OpReturn = 253;

    const auto valid = MinimalValidSpirv();

    auto untrusted_generator = valid;
    untrusted_generator[2] = 1;
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(untrusted_generator));

    auto no_shader_capability = valid;
    const auto capability = FindSpirvInstruction(no_shader_capability, OpCapability);
    ASSERT_TRUE(capability);
    no_shader_capability[*capability + 1] = 0;
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(no_shader_capability));

    auto no_memory_model = valid;
    ASSERT_TRUE(EraseSpirvInstruction(no_memory_model, OpMemoryModel));
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(no_memory_model));

    auto no_entry_point = valid;
    ASSERT_TRUE(EraseSpirvInstruction(no_entry_point, OpEntryPoint));
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(no_entry_point));

    auto unterminated_entry_point_name = valid;
    const auto entry_point = FindSpirvInstruction(unterminated_entry_point_name, OpEntryPoint);
    ASSERT_TRUE(entry_point);
    unterminated_entry_point_name[*entry_point + 4] = 0x41414141U;
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(unterminated_entry_point_name));

    auto missing_entry_point_function = valid;
    const auto function = FindSpirvInstruction(missing_entry_point_function, OpFunction);
    ASSERT_TRUE(function);
    missing_entry_point_function.erase(missing_entry_point_function.begin() + *function,
                                       missing_entry_point_function.end());
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(missing_entry_point_function));

    auto entry_point_is_label = valid;
    const auto mismatched_entry_point = FindSpirvInstruction(entry_point_is_label, OpEntryPoint);
    ASSERT_TRUE(mismatched_entry_point);
    entry_point_is_label[*mismatched_entry_point + 2] = 5;
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(entry_point_is_label));

    auto duplicate_result_id = valid;
    const auto label = FindSpirvInstruction(duplicate_result_id, OpLabel);
    ASSERT_TRUE(label);
    duplicate_result_id[*label + 1] = 4;
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(duplicate_result_id));

    auto unterminated_function = valid;
    ASSERT_TRUE(EraseSpirvInstruction(unterminated_function, OpReturn));
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(unterminated_function));

    auto unknown_opcode = valid;
    const auto return_instruction = FindSpirvInstruction(unknown_opcode, OpReturn);
    ASSERT_TRUE(return_instruction);
    unknown_opcode[*return_instruction] = 0x0001FFFEU;
    EXPECT_FALSE(Vulkan::IsValidCachedSpirv(unknown_opcode));
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
