// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/specialization_identity.h"

#include <algorithm>
#include <string_view>
#include <type_traits>
#include <vector>
#include <xxhash.h>

namespace Shader {
namespace {

class CanonicalWriter {
public:
    void WriteBool(bool value) {
        WriteU8(value ? 1 : 0);
    }

    void WriteU8(u8 value) {
        bytes.push_back(value);
    }

    void WriteS8(s8 value) {
        WriteU8(static_cast<u8>(value));
    }

    void WriteU32(u32 value) {
        for (u32 shift = 0; shift < 32; shift += 8) {
            WriteU8(static_cast<u8>(value >> shift));
        }
    }

    void WriteU64(u64 value) {
        for (u32 shift = 0; shift < 64; shift += 8) {
            WriteU8(static_cast<u8>(value >> shift));
        }
    }

    template <typename Enum>
        requires std::is_enum_v<Enum>
    void WriteEnum(Enum value) {
        WriteU64(static_cast<u64>(value));
    }

    void WriteCount(size_t count) {
        WriteU32(static_cast<u32>(count));
    }

    void WriteDomain(std::string_view domain) {
        WriteCount(domain.size());
        bytes.insert(bytes.end(), domain.begin(), domain.end());
    }

    [[nodiscard]] std::span<const u8> Bytes() const {
        return bytes;
    }

private:
    std::vector<u8> bytes;
};

void WriteCompMapping(CanonicalWriter& writer, AmdGpu::CompMapping mapping) {
    for (const auto component : mapping.array) {
        writer.WriteEnum(component);
    }
}

void WriteOutputMap(CanonicalWriter& writer, const OutputMap& outputs) {
    for (const auto output : outputs) {
        writer.WriteEnum(output);
    }
}

void WriteCommonRuntimeInfo(CanonicalWriter& writer, const RuntimeInfo& info) {
    writer.WriteEnum(info.stage);
    writer.WriteU32(info.num_user_data);
    writer.WriteU32(info.num_input_vgprs);
    writer.WriteU32(info.num_allocated_vgprs);
    writer.WriteEnum(info.fp_denorm_mode32);
    writer.WriteEnum(info.fp_denorm_mode16_64);
    writer.WriteEnum(info.fp_round_mode32);
    writer.WriteEnum(info.fp_round_mode16_64);
}

void WriteCommonHsEsVsInfo(CanonicalWriter& writer, const RuntimeInfo& info) {
    writer.WriteU32(info.hs_es_vs_info.hs_output_cp_stride);
}

void WriteCommonEsVsInfo(CanonicalWriter& writer, const RuntimeInfo& info) {
    WriteCommonHsEsVsInfo(writer, info);
    writer.WriteEnum(info.es_vs_info.tess_type);
    writer.WriteEnum(info.es_vs_info.tess_topology);
    writer.WriteEnum(info.es_vs_info.tess_partitioning);
}

void WritePsInputFlags(CanonicalWriter& writer, const AmdGpu::PsInput& input) {
    writer.WriteBool(input.persp_sample_ena);
    writer.WriteBool(input.persp_center_ena);
    writer.WriteBool(input.persp_centroid_ena);
    writer.WriteBool(input.persp_pull_model_ena);
    writer.WriteBool(input.linear_sample_ena);
    writer.WriteBool(input.linear_center_ena);
    writer.WriteBool(input.linear_centroid_ena);
    writer.WriteBool(input.line_stipple_tex_ena);
    writer.WriteBool(input.pos_x_float_ena);
    writer.WriteBool(input.pos_y_float_ena);
    writer.WriteBool(input.pos_z_float_ena);
    writer.WriteBool(input.pos_w_float_ena);
    writer.WriteBool(input.front_face_ena);
    writer.WriteBool(input.ancillary_ena);
    writer.WriteBool(input.sample_coverage_ena);
    writer.WriteBool(input.pos_fixed_pt_ena);
}

void WritePsColorBuffer(CanonicalWriter& writer, const PsColorBuffer& color) {
    writer.WriteEnum(color.data_format);
    writer.WriteEnum(color.num_format);
    writer.WriteEnum(color.num_conversion);
    writer.WriteEnum(color.export_format);
    writer.WriteBool(color.blend_self_scale);
    WriteCompMapping(writer, color.swizzle);
}

void WriteRuntimeInfo(CanonicalWriter& writer, const RuntimeInfo& info) {
    WriteCommonRuntimeInfo(writer, info);
    switch (info.stage) {
    case Stage::Local:
        writer.WriteU32(info.ls_info.ls_stride);
        break;
    case Stage::Export:
        WriteCommonEsVsInfo(writer, info);
        writer.WriteU32(info.es_info.vertex_data_size);
        break;
    case Stage::Vertex:
        WriteCommonEsVsInfo(writer, info);
        writer.WriteU32(info.vs_info.num_outputs);
        for (const auto& outputs : info.vs_info.outputs) {
            WriteOutputMap(writer, outputs);
        }
        writer.WriteBool(info.vs_info.tess_emulated_primitive);
        writer.WriteBool(info.vs_info.emulate_depth_negative_one_to_one);
        writer.WriteBool(info.vs_info.clip_disable);
        writer.WriteU32(info.vs_info.step_rate_0);
        writer.WriteU32(info.vs_info.step_rate_1);
        writer.WriteU32(info.vs_info.user_clip_plane_mask);
        writer.WriteU8(info.vs_info.draw_user_data.base_vertex);
        writer.WriteU8(info.vs_info.draw_user_data.base_instance);
        writer.WriteU8(info.vs_info.draw_user_data.draw_index);
        break;
    case Stage::Hull:
        WriteCommonHsEsVsInfo(writer, info);
        writer.WriteU32(info.hs_info.num_input_control_points);
        writer.WriteU32(info.hs_info.num_threads);
        writer.WriteEnum(info.hs_info.tess_type);
        writer.WriteBool(info.hs_info.offchip_lds_enable);
        writer.WriteU32(info.hs_info.ls_stride);
        writer.WriteU32(info.hs_info.hs_output_base);
        break;
    case Stage::Geometry:
        writer.WriteU32(info.gs_info.num_outputs);
        for (const auto& outputs : info.gs_info.outputs) {
            WriteOutputMap(writer, outputs);
        }
        writer.WriteU32(info.gs_info.num_invocations);
        writer.WriteU32(info.gs_info.output_vertices);
        writer.WriteU32(info.gs_info.in_vertex_data_size);
        writer.WriteU32(info.gs_info.out_vertex_data_size);
        writer.WriteEnum(info.gs_info.in_primitive);
        for (const auto primitive : info.gs_info.out_primitive) {
            writer.WriteEnum(primitive);
        }
        writer.WriteEnum(info.gs_info.mode);
        writer.WriteCount(info.gs_info.vs_copy.size());
        writer.WriteU64(info.gs_info.vs_copy_hash);
        for (const u32 instruction : info.gs_info.vs_copy) {
            writer.WriteU32(instruction);
        }
        break;
    case Stage::Fragment: {
        WritePsInputFlags(writer, info.fs_info.en_flags);
        WritePsInputFlags(writer, info.fs_info.addr_flags);
        writer.WriteU32(info.fs_info.num_inputs);
        const u32 num_inputs = std::min<u32>(info.fs_info.num_inputs, info.fs_info.inputs.size());
        for (u32 index = 0; index < num_inputs; ++index) {
            const auto& input = info.fs_info.inputs[index];
            writer.WriteU8(input.param_index);
            writer.WriteBool(input.is_default);
            writer.WriteBool(input.is_flat);
            writer.WriteU8(input.default_value);
        }
        for (const auto& color : info.fs_info.color_buffers) {
            WritePsColorBuffer(writer, color);
        }
        writer.WriteEnum(info.fs_info.z_export_format);
        writer.WriteU8(info.fs_info.mrtz_mask);
        writer.WriteBool(info.fs_info.dual_source_blending);
        writer.WriteBool(info.fs_info.clip_distance_emulation);
        break;
    }
    case Stage::Compute:
        writer.WriteU32(info.cs_info.shared_memory_size);
        for (const u32 size : info.cs_info.workgroup_size) {
            writer.WriteU32(size);
        }
        for (const bool enabled : info.cs_info.tgid_enable) {
            writer.WriteBool(enabled);
        }
        break;
    }
}

void WriteFetchShader(CanonicalWriter& writer,
                      const std::optional<Gcn::FetchShaderData>& fetch_shader) {
    writer.WriteBool(fetch_shader.has_value());
    if (!fetch_shader) {
        return;
    }
    writer.WriteS8(fetch_shader->vertex_offset_sgpr);
    writer.WriteS8(fetch_shader->instance_offset_sgpr);
    writer.WriteCount(fetch_shader->attributes.size());
    for (const auto& attribute : fetch_shader->attributes) {
        writer.WriteU8(attribute.semantic);
        writer.WriteU8(attribute.dest_vgpr);
        writer.WriteU8(attribute.num_elements);
        writer.WriteU8(attribute.sgpr_base);
        writer.WriteU8(attribute.dword_offset);
        writer.WriteU8(attribute.instance_data);
        writer.WriteU8(attribute.inst_offset);
        writer.WriteU8(attribute.data_format);
        writer.WriteU8(attribute.num_format);
    }
}

void WriteStageSpecialization(CanonicalWriter& writer, const StageSpecialization& spec) {
    WriteRuntimeInfo(writer, spec.runtime_info);

    const bool has_unified_resources =
        !spec.buffers.empty() || !spec.images.empty() || !spec.samplers.empty();
    writer.WriteBool(has_unified_resources);
    if (has_unified_resources) {
        writer.WriteU32(spec.start.unified);
    }
    const bool has_buffers = !spec.buffers.empty();
    writer.WriteBool(has_buffers);
    if (has_buffers) {
        writer.WriteU32(spec.start.buffer);
    }
    writer.WriteU32(spec.user_data_mask);
    const bool has_user_data = spec.user_data_mask != 0;
    writer.WriteBool(has_user_data);
    if (has_user_data) {
        writer.WriteU32(spec.start.user_data);
    }

    WriteFetchShader(writer, spec.fetch_shader_data);

    writer.WriteCount(spec.vs_attribs.size());
    for (const auto& attrib : spec.vs_attribs) {
        writer.WriteU32(attrib.divisor);
        writer.WriteEnum(attrib.num_class);
        WriteCompMapping(writer, attrib.dst_select);
    }

    writer.WriteCount(spec.buffers.size());
    for (const auto& buffer : spec.buffers) {
        writer.WriteU32(buffer.stride);
        writer.WriteBool(buffer.is_storage);
        writer.WriteBool(buffer.is_coherent);
        writer.WriteBool(buffer.is_formatted);
        writer.WriteBool(buffer.swizzle_enable);
        writer.WriteBool(buffer.add_tid_enable);
        if (buffer.is_formatted) {
            writer.WriteU32(buffer.data_format);
            writer.WriteU32(buffer.num_format);
            WriteCompMapping(writer, buffer.dst_select);
            writer.WriteEnum(buffer.num_conversion);
        }
        if (buffer.swizzle_enable) {
            writer.WriteU32(buffer.index_stride);
            writer.WriteU32(buffer.element_size);
        }
    }

    writer.WriteCount(spec.images.size());
    for (const auto& image : spec.images) {
        writer.WriteEnum(image.type);
        writer.WriteEnum(image.number_class);
        writer.WriteBool(image.is_storage);
        writer.WriteBool(image.is_cube);
        writer.WriteBool(image.is_srgb);
        WriteCompMapping(writer, image.dst_select);
        writer.WriteEnum(image.num_conversion);
        writer.WriteEnum(image.atomic_data_format);
        writer.WriteEnum(image.atomic_number_format);
        writer.WriteU32(image.num_bindings);
        writer.WriteU32(image.dynamic_descriptor_count);
        writer.WriteU32(image.dynamic_descriptor_stride_dw);
        writer.WriteBool(image.requires_dynamic_oob_guard);
    }

    writer.WriteCount(spec.fmasks.size());
    for (const auto& fmask : spec.fmasks) {
        writer.WriteU32(fmask.width);
        writer.WriteU32(fmask.height);
    }

    writer.WriteCount(spec.samplers.size());
    for (const auto& sampler : spec.samplers) {
        writer.WriteBool(sampler.force_unnormalized);
        writer.WriteBool(sampler.force_degamma);
    }
}

} // namespace

u64 ComputeShaderPermutationIdentity(u64 program_hash, LogicalStage logical_stage,
                                     const StageSpecialization& spec) {
    CanonicalWriter writer;
    writer.WriteDomain("shadPS4.shader-permutation-identity");
    writer.WriteU32(5); // Canonical identity schema: dynamic draw user-data destinations.
    writer.WriteU64(program_hash);
    writer.WriteEnum(logical_stage);
    WriteStageSpecialization(writer, spec);

    const auto bytes = writer.Bytes();
    const u64 identity = XXH3_64bits(bytes.data(), bytes.size());
    // Zero denotes an absent stage in GraphicsPipelineKey.
    return identity != 0 ? identity : 0x9e3779b97f4a7c15ULL;
}

} // namespace Shader
