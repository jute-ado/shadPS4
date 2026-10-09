// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/vk_pipeline_cache_serialization.h"

#include <array>
#include <bit>
#include <cmath>
#include <ranges>

#include "common/serdes.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {
namespace {

[[nodiscard]] constexpr bool IsValidPrimitiveType(AmdGpu::PrimitiveType type) noexcept {
    using enum AmdGpu::PrimitiveType;
    switch (type) {
    case PointList:
    case LineList:
    case LineStrip:
    case TriangleList:
    case TriangleFan:
    case TriangleStrip:
    case PatchPrimitive:
    case AdjLineList:
    case AdjLineStrip:
    case AdjTriangleList:
    case AdjTriangleStrip:
    case RectList:
    case QuadList:
    case Polygon:
        return true;
    case None:
    case LineLoop:
    case QuadStrip:
        return false;
    }
    return false;
}

[[nodiscard]] constexpr bool IsValidBlendFactor(AmdGpu::BlendControl::BlendFactor factor) noexcept {
    using enum AmdGpu::BlendControl::BlendFactor;
    switch (factor) {
    case Zero:
    case One:
    case SrcColor:
    case OneMinusSrcColor:
    case SrcAlpha:
    case OneMinusSrcAlpha:
    case DstAlpha:
    case OneMinusDstAlpha:
    case DstColor:
    case OneMinusDstColor:
    case SrcAlphaSaturate:
    case ConstantColor:
    case OneMinusConstantColor:
    case Src1Color:
    case InvSrc1Color:
    case Src1Alpha:
    case InvSrc1Alpha:
    case ConstantAlpha:
    case OneMinusConstantAlpha:
        return true;
    }
    return false;
}

[[nodiscard]] constexpr bool IsValidBlendControl(const AmdGpu::BlendControl& control) noexcept {
    return IsValidBlendFactor(control.color_src_factor) &&
           IsValidBlendFactor(control.color_dst_factor) &&
           IsValidBlendFactor(control.alpha_src_factor) &&
           IsValidBlendFactor(control.alpha_dst_factor) &&
           static_cast<u32>(control.color_func) <=
               static_cast<u32>(AmdGpu::BlendControl::BlendFunc::ReverseSubtract) &&
           static_cast<u32>(control.alpha_func) <=
               static_cast<u32>(AmdGpu::BlendControl::BlendFunc::ReverseSubtract);
}

[[nodiscard]] constexpr bool IsValidLogicOp(AmdGpu::ColorControl::LogicOp op) noexcept {
    const u32 value = static_cast<u32>(op);
    return value <= 0xFF && (value & 0xFU) == (value >> 4);
}

[[nodiscard]] constexpr bool IsValidNumberFormat(AmdGpu::NumberFormat format) noexcept {
    using enum AmdGpu::NumberFormat;
    switch (format) {
    case Unorm:
    case Snorm:
    case Uscaled:
    case Sscaled:
    case Uint:
    case Sint:
    case SnormNz:
    case Float:
    case Srgb:
    case Ubnorm:
    case UbnormNz:
    case Ubint:
    case Ubscaled:
        return true;
    }
    return false;
}

[[nodiscard]] constexpr bool IsSupportedColorFormat(AmdGpu::DataFormat data,
                                                    AmdGpu::NumberFormat number) noexcept {
    using enum AmdGpu::DataFormat;
    using enum AmdGpu::NumberFormat;
    if (!IsValidNumberFormat(number)) {
        return false;
    }
    switch (data) {
    case FormatInvalid:
        return true;
    case Format8:
    case Format8_8:
    case Format8_8_8_8:
        return number == Unorm || number == Snorm || number == Uint || number == Sint ||
               number == Srgb;
    case Format16:
    case Format16_16:
        return number == Unorm || number == Snorm || number == Uint || number == Sint ||
               number == Float;
    case Format32:
    case Format32_32:
    case Format32_32_32:
    case Format32_32_32_32:
        return number == Uint || number == Sint || number == Float;
    case Format10_11_11:
    case Format5_9_9_9:
        return number == Float;
    case Format2_10_10_10:
        return number == Unorm || number == Snorm || number == Uint || number == Sint;
    case Format16_16_16_16:
        return number == Unorm || number == Snorm || number == SnormNz || number == Uint ||
               number == Sint || number == Float;
    case Format5_6_5:
    case Format1_5_5_5:
    case Format5_5_5_1:
    case Format4_4_4_4:
        return number == Unorm;
    case FormatBc1:
    case FormatBc2:
    case FormatBc3:
    case FormatBc7:
        return number == Unorm || number == Srgb;
    case FormatBc4:
    case FormatBc5:
    case FormatBc6:
        return number == Unorm || number == Snorm;
    default:
        return false;
    }
}

[[nodiscard]] constexpr bool IsValidCompMapping(AmdGpu::CompMapping mapping) noexcept {
    return std::ranges::all_of(mapping.array, [](AmdGpu::CompSwizzle component) {
        const u32 value = static_cast<u32>(component);
        return value <= 1 || (value >= 4 && value <= 7);
    });
}

[[nodiscard]] constexpr bool IsValidSampleCount(u32 samples, bool allow_zero = false) noexcept {
    return (allow_zero && samples == 0) ||
           (samples >= 1 && samples <= 16 && std::has_single_bit(samples));
}

[[nodiscard]] constexpr bool IsCoreFormat(vk::Format format, bool allow_undefined) noexcept {
    const auto value = static_cast<std::underlying_type_t<vk::Format>>(format);
    const auto last =
        static_cast<std::underlying_type_t<vk::Format>>(vk::Format::eAstc12x12SrgbBlock);
    return value >= (allow_undefined ? 0 : 1) && value <= last;
}

[[nodiscard]] bool ValidateVertexInputs(
    const GraphicsPipeline::SerializationSupport& support) noexcept {
    if (support.vertex_attributes.size() != support.vertex_bindings.size() ||
        support.divisors.size() > support.vertex_bindings.size()) {
        return false;
    }

    std::array<bool, MaxVertexBufferCount> bindings{};
    std::array<bool, MaxVertexBufferCount> locations{};
    for (const auto& binding : support.vertex_bindings) {
        if (binding.binding >= MaxVertexBufferCount || bindings[binding.binding] ||
            (binding.inputRate != vk::VertexInputRate::eVertex &&
             binding.inputRate != vk::VertexInputRate::eInstance)) {
            return false;
        }
        bindings[binding.binding] = true;
    }
    for (const auto& attribute : support.vertex_attributes) {
        if (attribute.location >= MaxVertexBufferCount ||
            attribute.binding >= MaxVertexBufferCount || locations[attribute.location] ||
            !bindings[attribute.binding] || !IsCoreFormat(attribute.format, false)) {
            return false;
        }
        locations[attribute.location] = true;
    }
    std::array<bool, MaxVertexBufferCount> divisor_bindings{};
    for (const auto& divisor : support.divisors) {
        if (divisor.binding >= MaxVertexBufferCount || !bindings[divisor.binding] ||
            divisor_bindings[divisor.binding]) {
            return false;
        }
        divisor_bindings[divisor.binding] = true;
    }
    return true;
}

[[nodiscard]] GraphicsPipelineKey CanonicalizeGraphicsPipelineKey(
    const GraphicsPipelineKey& key) noexcept {
    GraphicsPipelineKey canonical{};
    canonical.stage_hashes = key.stage_hashes;
    canonical.vertex_buffer_formats = key.vertex_buffer_formats;
    canonical.patch_control_points = key.patch_control_points;
    canonical.num_color_attachments = key.num_color_attachments;
    for (size_t index = 0; index < AmdGpu::NUM_COLOR_BUFFERS; ++index) {
        const auto& color = key.color_buffers[index];
        auto& canonical_color = canonical.color_buffers[index];
        canonical_color.data_format = color.data_format;
        canonical_color.num_format = color.num_format;
        canonical_color.num_conversion = color.num_conversion;
        canonical_color.export_format = color.export_format;
        canonical_color.blend_self_scale = color.blend_self_scale;
        canonical_color.swizzle.r = color.swizzle.r;
        canonical_color.swizzle.g = color.swizzle.g;
        canonical_color.swizzle.b = color.swizzle.b;
        canonical_color.swizzle.a = color.swizzle.a;

        const auto& blend = key.blend_controls[index];
        auto& canonical_blend = canonical.blend_controls[index];
        canonical_blend.color_src_factor = blend.color_src_factor;
        canonical_blend.color_func = blend.color_func;
        canonical_blend.color_dst_factor = blend.color_dst_factor;
        canonical_blend.alpha_src_factor = blend.alpha_src_factor;
        canonical_blend.alpha_func = blend.alpha_func;
        canonical_blend.alpha_dst_factor = blend.alpha_dst_factor;
        canonical_blend.separate_alpha_blend = blend.separate_alpha_blend;
        canonical_blend.enable = blend.enable;
        canonical_blend.disable_rop3 = blend.disable_rop3;

        using WriteMask = vk::ColorComponentFlags::MaskType;
        canonical.write_masks[index] =
            vk::ColorComponentFlags{static_cast<WriteMask>(key.write_masks[index])};
    }
    canonical.cb_shader_mask.raw = key.cb_shader_mask.raw;
    canonical.logic_op = key.logic_op;
    canonical.num_samples = key.num_samples;
    canonical.depth_samples = key.depth_samples;
    canonical.color_samples = key.color_samples;
    canonical.mrt_mask = key.mrt_mask;
    canonical.z_format = key.z_format;
    canonical.stencil_format = key.stencil_format;
    canonical.depth_clamp_enable = key.depth_clamp_enable;
    canonical.prim_type = key.prim_type;
    canonical.polygon_mode = key.polygon_mode;
    canonical.clip_space = key.clip_space;
    canonical.provoking_vtx_last = key.provoking_vtx_last;
    canonical.depth_clip_enable = key.depth_clip_enable;
    return canonical;
}

} // namespace

bool ValidateGraphicsPipelineKey(const GraphicsPipelineKey& key) noexcept {
    const auto canonical = CanonicalizeGraphicsPipelineKey(key);
    if (std::memcmp(&key, &canonical, sizeof(key)) != 0) {
        return false;
    }
    const auto compute_slot = static_cast<u32>(Shader::LogicalStage::Compute);
    const auto vertex_slot = static_cast<u32>(Shader::LogicalStage::Vertex);
    const auto tcs_slot = static_cast<u32>(Shader::LogicalStage::TessellationControl);
    const auto tes_slot = static_cast<u32>(Shader::LogicalStage::TessellationEval);
    const bool has_tcs = key.stage_hashes[tcs_slot] != 0;
    const bool has_tes = key.stage_hashes[tes_slot] != 0;
    if (key.stage_hashes[compute_slot] != 0 || key.stage_hashes[vertex_slot] == 0 ||
        has_tcs != has_tes || !IsValidPrimitiveType(key.prim_type) ||
        static_cast<u32>(key.polygon_mode) > static_cast<u32>(AmdGpu::PolygonMode::Fill) ||
        static_cast<u32>(key.clip_space) > static_cast<u32>(AmdGpu::ClipSpace::ZeroToW) ||
        static_cast<u32>(key.provoking_vtx_last) >
            static_cast<u32>(AmdGpu::ProvokingVtxLast::Last) ||
        !IsValidLogicOp(key.logic_op) || !IsValidSampleCount(key.num_samples) ||
        !IsValidSampleCount(key.depth_samples) ||
        (key.z_format != AmdGpu::DepthBuffer::ZFormat::Invalid &&
         key.z_format != AmdGpu::DepthBuffer::ZFormat::Z16 &&
         key.z_format != AmdGpu::DepthBuffer::ZFormat::Z32Float) ||
        static_cast<u32>(key.stencil_format) >
            static_cast<u32>(AmdGpu::DepthBuffer::StencilFormat::Stencil8) ||
        key.num_color_attachments > AmdGpu::NUM_COLOR_BUFFERS ||
        (key.mrt_mask >> AmdGpu::NUM_COLOR_BUFFERS) != 0 ||
        key.num_color_attachments != std::bit_width(key.mrt_mask) ||
        key.patch_control_points > 32) {
        return false;
    }

    const bool is_patch = key.prim_type == AmdGpu::PrimitiveType::PatchPrimitive;
    if ((is_patch && (!has_tcs || key.patch_control_points == 0)) ||
        (!is_patch && key.patch_control_points != 0)) {
        return false;
    }

    u32 maximum_samples = key.depth_samples;
    for (size_t index = 0; index < AmdGpu::NUM_COLOR_BUFFERS; ++index) {
        const auto& color = key.color_buffers[index];
        if (!IsSupportedColorFormat(color.data_format, color.num_format) ||
            static_cast<u32>(color.num_conversion) >
                static_cast<u32>(AmdGpu::NumberConversion::SrgbToNorm) ||
            static_cast<u32>(color.export_format) >
                static_cast<u32>(AmdGpu::ShaderExportFormat::ABGR_32) ||
            !IsValidCompMapping(color.swizzle) || !IsValidBlendControl(key.blend_controls[index]) ||
            (static_cast<u32>(key.write_masks[index]) & ~0xFU) != 0 ||
            !IsValidSampleCount(key.color_samples[index], true)) {
            return false;
        }
        maximum_samples = std::max(maximum_samples, static_cast<u32>(key.color_samples[index]));
    }
    if (key.num_samples != maximum_samples) {
        return false;
    }
    return std::ranges::all_of(key.vertex_buffer_formats,
                               [](vk::Format format) { return IsCoreFormat(format, true); });
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};
    const auto canonical = CanonicalizeGraphicsPipelineKey(*this);
    key.Write(&canonical, sizeof(canonical));
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader reader{ar};
    GraphicsPipelineKey decoded{};
    if (!reader.Read(&decoded, sizeof(decoded)) || !reader.IsValid() ||
        !ValidateGraphicsPipelineKey(decoded)) {
        return false;
    }
    *this = decoded;
    return true;
}

void GraphicsPipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer support{ar};
    support.Write(vertex_attributes);
    support.Write(vertex_bindings);
    support.Write(divisors);
    support.Write(multisampling.flags);
    support.Write(multisampling.rasterizationSamples);
    support.Write(multisampling.sampleShadingEnable);
    support.Write(multisampling.minSampleShading);
    support.Write(multisampling.alphaToCoverageEnable);
    support.Write(multisampling.alphaToOneEnable);
    support.Write(tcs);
    support.Write(tes);
    support.Write(fragment);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader reader{ar};
    SerializationSupport decoded{};
    vk::PipelineCreateFlags flags{};
    vk::SampleCountFlagBits rasterization_samples{};
    vk::Bool32 sample_shading_enable{};
    float min_sample_shading{};
    vk::Bool32 alpha_to_coverage_enable{};
    vk::Bool32 alpha_to_one_enable{};
    if (!reader.Read(decoded.vertex_attributes, MaxVertexBufferCount) ||
        !reader.Read(decoded.vertex_bindings, MaxVertexBufferCount) ||
        !reader.Read(decoded.divisors, MaxVertexBufferCount) || !reader.Read(flags) ||
        !reader.Read(rasterization_samples) || !reader.Read(sample_shading_enable) ||
        !reader.Read(min_sample_shading) || !reader.Read(alpha_to_coverage_enable) ||
        !reader.Read(alpha_to_one_enable) || !reader.Read(decoded.tcs, MaxCachedShaderDwords) ||
        !reader.Read(decoded.tes, MaxCachedShaderDwords) ||
        !reader.Read(decoded.fragment, MaxCachedShaderDwords)) {
        return false;
    }
    const u32 sample_count = static_cast<u32>(rasterization_samples);
    if (!reader.IsValid() || flags || sample_count == 0 || sample_count > 16 ||
        !std::has_single_bit(sample_count) || sample_shading_enable > 1 ||
        alpha_to_coverage_enable > 1 || alpha_to_one_enable > 1 ||
        !std::isfinite(min_sample_shading) || min_sample_shading < 0.0f ||
        min_sample_shading > 1.0f || (!decoded.tcs.empty() && !IsValidCachedSpirv(decoded.tcs)) ||
        (!decoded.tes.empty() && !IsValidCachedSpirv(decoded.tes)) ||
        (!decoded.fragment.empty() && !IsValidCachedSpirv(decoded.fragment))) {
        return false;
    }
    decoded.multisampling = {
        .rasterizationSamples = rasterization_samples,
        .sampleShadingEnable = sample_shading_enable,
        .minSampleShading = min_sample_shading,
        .alphaToCoverageEnable = alpha_to_coverage_enable,
        .alphaToOneEnable = alpha_to_one_enable,
    };
    if (!ValidateVertexInputs(decoded)) {
        return false;
    }
    *this = std::move(decoded);
    return true;
}

bool DeserializeGraphicsPipelineCacheRecord(Serialization::Archive& ar, GraphicsPipelineKey& key,
                                            GraphicsPipeline::SerializationSupport& support) {
    GraphicsPipelineKey decoded_key{};
    GraphicsPipeline::SerializationSupport decoded_support{};
    if (!decoded_key.Deserialize(ar) || !decoded_support.Deserialize(ar) || !ar.IsEoS()) {
        return false;
    }
    key = decoded_key;
    support = std::move(decoded_support);
    return true;
}

} // namespace Vulkan
