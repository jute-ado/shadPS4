// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/static_vector.hpp>
#include "common/logging/log.h"
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_image_coordinates.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

struct ImageOperands {
    void Add(spv::ImageOperandsMask new_mask, Id value) {
        if (!Sirit::ValidId(value)) {
            return;
        }
        mask = static_cast<spv::ImageOperandsMask>(static_cast<u32>(mask) |
                                                   static_cast<u32>(new_mask));
        operands.push_back(value);
    }
    void Add(spv::ImageOperandsMask new_mask, Id value1, Id value2) {
        mask = static_cast<spv::ImageOperandsMask>(static_cast<u32>(mask) |
                                                   static_cast<u32>(new_mask));
        operands.push_back(value1);
        operands.push_back(value2);
    }

    void AddOffset(EmitContext& ctx, const IR::Value& offset,
                   bool can_use_runtime_offsets = false) {
        if (offset.IsEmpty()) {
            return;
        }
        if (offset.IsImmediate()) {
            const s32 operand = offset.U32();
            Add(spv::ImageOperandsMask::ConstOffset, ctx.ConstS32(operand));
            return;
        }
        IR::Inst* const inst{offset.InstRecursive()};
        if (inst->AreAllArgsImmediates()) {
            switch (inst->GetOpcode()) {
            case IR::Opcode::CompositeConstructU32x2:
                Add(spv::ImageOperandsMask::ConstOffset,
                    ctx.ConstS32(static_cast<s32>(inst->Arg(0).U32()),
                                 static_cast<s32>(inst->Arg(1).U32())));
                return;
            case IR::Opcode::CompositeConstructU32x3:
                Add(spv::ImageOperandsMask::ConstOffset,
                    ctx.ConstS32(static_cast<s32>(inst->Arg(0).U32()),
                                 static_cast<s32>(inst->Arg(1).U32()),
                                 static_cast<s32>(inst->Arg(2).U32())));
                return;
            default:
                break;
            }
        }
        if (can_use_runtime_offsets) {
            Add(spv::ImageOperandsMask::Offset, ctx.Def(offset));
        } else {
            LOG_WARNING(Render_Vulkan,
                        "Runtime offset provided to unsupported image sample instruction");
        }
    }

    void AddDerivatives(EmitContext& ctx, Id derivatives_dx, Id derivatives_dy) {
        if (!Sirit::ValidId(derivatives_dx) || !Sirit::ValidId(derivatives_dy)) {
            return;
        }
        Add(spv::ImageOperandsMask::Grad, derivatives_dx, derivatives_dy);
    }

    spv::ImageOperandsMask mask{};
    boost::container::static_vector<Id, 4> operands;
};

[[nodiscard]] bool ValidateImageCoordinates(EmitContext& ctx, const IR::Inst& inst,
                                            const EmitContext::TextureDefinition& texture,
                                            u32 coordinate_arg, bool integer_coordinates) {
    const IR::Type coordinate_type = inst.Arg(coordinate_arg).Type();
    if (IsImageCoordinateTypeValid(texture.view_type, coordinate_type, integer_coordinates)) {
        return true;
    }

    LOG_WARNING(Render_Recompiler,
                "Rejecting image operation {} for shader {:#x}: view type {} requires at least "
                "{} {} coordinate components, but IR supplied {} ({})",
                u32(inst.GetOpcode()), ctx.info.pgm_hash, AmdGpu::NameOf(texture.view_type),
                integer_coordinates ? "32-bit unsigned integer" : "32-bit floating-point",
                ImageCoordinateComponentCount(texture.view_type),
                CoordinateComponentCount(coordinate_type), IR::NameOf(coordinate_type));
    // Do not synthesize a missing coordinate component. A guessed component can address a
    // different texel or array slice; keeping the operation out of SPIR-V is safer than emitting
    // a module that Vulkan rejects. The typed zero keeps the surrounding IR well-formed while the
    // owner records the shader as rejected.
    ctx.info.translation_failed = true;
    return false;
}

[[nodiscard]] bool ValidateImageSampleOperation(EmitContext& ctx, const IR::Inst& inst,
                                                const EmitContext::TextureDefinition& texture,
                                                u32 coordinate_arg) {
    if (!IsImageSampleOperationAllowed(texture.view_type)) {
        LOG_WARNING(Render_Recompiler,
                    "Rejecting sampled-image operation {} for shader {:#x}: multisampled view "
                    "{} requires image fetch/read with an explicit sample",
                    u32(inst.GetOpcode()), ctx.info.pgm_hash, AmdGpu::NameOf(texture.view_type));
        ctx.info.translation_failed = true;
        return false;
    }
    return ValidateImageCoordinates(ctx, inst, texture, coordinate_arg, false);
}

void AddImageSampleOperand(EmitContext& ctx, const EmitContext::TextureDefinition& texture,
                           Id image, Id ms, ImageOperands& operands, bool wrap_msaa_index) {
    if (!IsImageMultisampled(texture.view_type)) {
        if (Sirit::ValidId(ms)) {
            LOG_ERROR(Render_Recompiler, "image is not MS but ms operand is provided");
        }
        return;
    }
    if (!Sirit::ValidId(ms)) {
        return;
    }
    if (!wrap_msaa_index) {
        operands.Add(spv::ImageOperandsMask::Sample, ms);
        return;
    }

    // GCN hardware wraps out-of-range MSAA sample indices. Keep that behavior while ensuring
    // only multisampled SPIR-V image types receive the Sample operand.
    const Id sample_count = ctx.OpImageQuerySamples(ctx.U32[1], image);
    const Id wrapped_ms = ctx.OpUMod(ctx.U32[1], ms, sample_count);
    operands.Add(spv::ImageOperandsMask::Sample, wrapped_ms);
}

[[nodiscard]] Id RejectedImageSample(EmitContext& ctx) {
    return ctx.ConstF32(0.0f, 0.0f, 0.0f, 0.0f);
}

[[nodiscard]] Id RejectedImageQueryLod(EmitContext& ctx) {
    // ImageQueryLod is represented as the standard four-component image result in IR; the
    // backend's valid path fills the first two lanes with the SPIR-V query result.
    return ctx.ConstF32(0.0f, 0.0f, 0.0f, 0.0f);
}

Id EmitImageSampleRaw(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address1, Id address2,
                      Id address3, Id address4) {
    UNREACHABLE_MSG("Unreachable instruction");
}

Id EmitImageSampleImplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id bias,
                              const IR::Value& offset) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    if (!ValidateImageSampleOperation(ctx, *inst, texture, 1U)) {
        return RejectedImageSample(ctx);
    }
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Bias, bias);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleImplicitLod(result_type, sampled_image, coords,
                                                   operands.mask, operands.operands);
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], sample) : sample;
}

Id EmitImageSampleExplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id lod,
                              const IR::Value& offset) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    if (!ValidateImageSampleOperation(ctx, *inst, texture, 1U)) {
        return RejectedImageSample(ctx);
    }
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Lod, lod);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleExplicitLod(result_type, sampled_image, coords,
                                                   operands.mask, operands.operands);
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], sample) : sample;
}

Id EmitImageSampleExplicitLodDynamic(EmitContext& ctx, IR::Inst* inst, u32 handle,
                                     Id descriptor_index, Id coords, Id lod,
                                     const IR::Value& offset) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    if (!ValidateImageSampleOperation(ctx, *inst, texture, 2U)) {
        return RejectedImageSample(ctx);
    }
    ASSERT(texture.is_dynamic_descriptor_table && texture.descriptor_count != 0);

    const Id in_bounds_label = ctx.OpLabel();
    const Id out_of_bounds_label = ctx.OpLabel();
    const Id merge_label = ctx.OpLabel();
    const Id in_bounds =
        ctx.OpULessThan(ctx.U1[1], descriptor_index, ctx.ConstU32(texture.descriptor_count));
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(in_bounds, in_bounds_label, out_of_bounds_label);

    // Give this descriptor use its own identity. Decorating a shared guest index would incorrectly
    // mark every unrelated use of the same SSA value as nonuniform.
    ctx.AddLabel(in_bounds_label);
    const Id nonuniform_index = ctx.OpIAdd(ctx.U32[1], descriptor_index, ctx.u32_zero_value);
    ctx.Decorate(nonuniform_index, spv::Decoration::NonUniform);
    const Id image_pointer =
        ctx.OpAccessChain(texture.element_pointer_type, texture.id, nonuniform_index);
    ctx.Decorate(image_pointer, spv::Decoration::NonUniform);
    const Id image = ctx.OpLoad(texture.image_type, image_pointer);
    ctx.Decorate(image, spv::Decoration::NonUniform);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    // NonUniform must decorate the final operand consumed by the sampling instruction.
    ctx.Decorate(sampled_image, spv::Decoration::NonUniform);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Lod, lod);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleExplicitLod(result_type, sampled_image, coords,
                                                   operands.mask, operands.operands);
    const Id typed_sample = texture.is_integer ? ctx.OpBitcast(ctx.F32[4], sample) : sample;
    ctx.OpBranch(merge_label);

    // Never form a descriptor access on the OOB path. The bounded window is host capacity, not a
    // guessed guest-table length, so unsupported indices deterministically sample zero.
    ctx.AddLabel(out_of_bounds_label);
    const Id zero = ctx.ConstF32(0.0f, 0.0f, 0.0f, 0.0f);
    ctx.OpBranch(merge_label);

    ctx.AddLabel(merge_label);
    return ctx.OpPhi(ctx.F32[4], typed_sample, in_bounds_label, zero, out_of_bounds_label);
}

Id EmitImageSampleDrefImplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id dref,
                                  Id bias, const IR::Value& offset) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    if (!ValidateImageSampleOperation(ctx, *inst, texture, 1U)) {
        return RejectedImageSample(ctx);
    }
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id result_type = texture.data_types->Get(1);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Bias, bias);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleDrefImplicitLod(result_type, sampled_image, coords, dref,
                                                       operands.mask, operands.operands);
    const Id sample_typed = texture.is_integer ? ctx.OpBitcast(ctx.F32[1], sample) : sample;
    return ctx.OpCompositeConstruct(ctx.F32[4], sample_typed, ctx.f32_zero_value,
                                    ctx.f32_zero_value, ctx.f32_zero_value);
}

Id EmitImageSampleDrefExplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id dref,
                                  Id lod, const IR::Value& offset) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    if (!ValidateImageSampleOperation(ctx, *inst, texture, 1U)) {
        return RejectedImageSample(ctx);
    }
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id result_type = texture.data_types->Get(1);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Lod, lod);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleDrefExplicitLod(result_type, sampled_image, coords, dref,
                                                       operands.mask, operands.operands);
    const Id sample_typed = texture.is_integer ? ctx.OpBitcast(ctx.F32[1], sample) : sample;
    return ctx.OpCompositeConstruct(ctx.F32[4], sample_typed, ctx.f32_zero_value,
                                    ctx.f32_zero_value, ctx.f32_zero_value);
}

Id EmitImageGather(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords,
                   const IR::Value& offset) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    if (!ValidateImageSampleOperation(ctx, *inst, texture, 1U)) {
        return RejectedImageSample(ctx);
    }
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    const u32 comp = inst->Flags<IR::TextureInstInfo>().gather_comp.Value();
    ImageOperands operands;
    operands.AddOffset(ctx, offset, true);
    const Id texels = ctx.OpImageGather(result_type, sampled_image, coords, ctx.ConstU32(comp),
                                        operands.mask, operands.operands);
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], texels) : texels;
}

Id EmitImageGatherDref(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords,
                       const IR::Value& offset, Id dref) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    if (!ValidateImageSampleOperation(ctx, *inst, texture, 1U)) {
        return RejectedImageSample(ctx);
    }
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    ImageOperands operands;
    operands.AddOffset(ctx, offset, true);
    const Id texels = ctx.OpImageDrefGather(result_type, sampled_image, coords, dref, operands.mask,
                                            operands.operands);
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], texels) : texels;
}

Id EmitImageQueryDimensions(EmitContext& ctx, IR::Inst* inst, u32 handle, Id lod, bool has_mips) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const auto sharp = ctx.info.images[handle & 0xFFFF].GetSharp(ctx.info);
    const Id zero = ctx.u32_zero_value;
    const auto mips{[&] { return has_mips ? ctx.OpImageQueryLevels(ctx.U32[1], image) : zero; }};
    const bool uses_lod{!IsImageMultisampled(texture.view_type) && !texture.is_storage};
    const auto query{[&](Id type) {
        return uses_lod ? ctx.OpImageQuerySizeLod(type, image, lod)
                        : ctx.OpImageQuerySize(type, image);
    }};
    switch (texture.view_type) {
    case AmdGpu::ImageType::Color1D:
        return ctx.OpCompositeConstruct(ctx.U32[4], query(ctx.U32[1]), zero, zero, mips());
    case AmdGpu::ImageType::Color1DArray:
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
        return ctx.OpCompositeConstruct(ctx.U32[4], query(ctx.U32[2]), zero, mips());
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Color2DMsaaArray:
    case AmdGpu::ImageType::Color3D:
        return ctx.OpCompositeConstruct(ctx.U32[4], query(ctx.U32[3]), mips());
    default:
        UNREACHABLE_MSG("SPIR-V Instruction");
    }
}

Id EmitImageQueryLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    if (!ValidateImageSampleOperation(ctx, *inst, texture, 1U)) {
        return RejectedImageQueryLod(ctx);
    }
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    const Id query = ctx.OpImageQueryLod(ctx.F32[2], sampled_image, coords);
    // OpImageQueryLod natively returns vec2, while the IR opcode deliberately exposes the
    // standard four-component image-result shape.  Materialize that adapter here so later
    // component extraction (including lanes 2/3) cannot address a two-component SPIR-V value.
    const Id zero{ctx.f32_zero_value};
    return ctx.OpCompositeConstruct(ctx.F32[4], query, zero, zero);
}

Id EmitImageGradient(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id derivatives_dx,
                     Id derivatives_dy, const IR::Value& offset, const IR::Value& lod_clamp) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    if (!ValidateImageSampleOperation(ctx, *inst, texture, 1U)) {
        return RejectedImageSample(ctx);
    }
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    ImageOperands operands;
    operands.AddDerivatives(ctx, derivatives_dx, derivatives_dy);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleExplicitLod(result_type, sampled_image, coords,
                                                   operands.mask, operands.operands);
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], sample) : sample;
}

Id EmitImageRead(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id lod, Id ms) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    if (!ValidateImageCoordinates(ctx, *inst, texture, 1U, true)) {
        return RejectedImageSample(ctx);
    }
    const Id color_type = texture.data_types->Get(4);
    ImageOperands operands;
    Id texel;
    if (!texture.is_storage) {
        const Id image = ctx.OpLoad(texture.image_type, texture.id);
        AddImageSampleOperand(ctx, texture, image, ms, operands, true);
        if (!IsImageMultisampled(texture.view_type)) {
            operands.Add(spv::ImageOperandsMask::Lod, lod);
        }
        texel = ctx.OpImageFetch(color_type, image, coords, operands.mask, operands.operands);
    } else {
        Id image_ptr = texture.id;
        if (ctx.profile.supports_image_load_store_lod) {
            operands.Add(spv::ImageOperandsMask::Lod, lod);
        } else if (Sirit::ValidId(lod)) {
#if 1
            // It's confusing what interactions will cause this code path so leave it as
            // unreachable until a case is found.
            // Normally IMAGE_LOAD_MIP should translate -> OpImageFetch
            UNREACHABLE_MSG("Unsupported ImageRead with Lod");
#else
            LOG_WARNING(Render, "Fallback for ImageRead with LOD");
            ASSERT(texture.mip_fallback_mode == MipStorageFallbackMode::DynamicIndex);
            const Id single_image_ptr_type =
                ctx.TypePointer(spv::StorageClass::UniformConstant, texture.image_type);
            image_ptr = ctx.OpAccessChain(single_image_ptr_type, image_ptr, std::array{lod});
#endif
        }
        const Id image = ctx.OpLoad(texture.image_type, image_ptr);
        AddImageSampleOperand(ctx, texture, image, ms, operands, true);
        texel = ctx.OpImageRead(color_type, image, coords, operands.mask, operands.operands);
    }
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], texel) : texel;
}

void EmitImageWrite(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id lod, Id ms,
                    Id color) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    if (!ValidateImageCoordinates(ctx, *inst, texture, 1U, true)) {
        return;
    }
    Id image_ptr = texture.id;
    const Id color_type = texture.data_types->Get(4);
    ImageOperands operands;
    if (ctx.profile.supports_image_load_store_lod) {
        operands.Add(spv::ImageOperandsMask::Lod, lod);
    } else if (Sirit::ValidId(lod)) {
        LOG_WARNING(Render, "Fallback for ImageWrite with LOD");
        ASSERT(texture.mip_fallback_mode == MipStorageFallbackMode::DynamicIndex);
        const Id single_image_ptr_type =
            ctx.TypePointer(spv::StorageClass::UniformConstant, texture.image_type);
        image_ptr = ctx.OpAccessChain(single_image_ptr_type, image_ptr, std::array{lod});
    }
    const Id image = ctx.OpLoad(texture.image_type, image_ptr);
    AddImageSampleOperand(ctx, texture, image, ms, operands, false);
    const Id texel = texture.is_integer ? ctx.OpBitcast(color_type, color) : color;
    ctx.OpImageWrite(image, coords, texel, operands.mask, operands.operands);
}

Id EmitCubeFaceIndex(EmitContext& ctx, IR::Inst* inst, Id cube_coords) {
    if (ctx.profile.supports_native_cube_calc) {
        return ctx.OpCubeFaceIndexAMD(ctx.F32[1], cube_coords);
    } else {
        UNREACHABLE_MSG("SPIR-V Instruction");
    }
}

} // namespace Shader::Backend::SPIRV
