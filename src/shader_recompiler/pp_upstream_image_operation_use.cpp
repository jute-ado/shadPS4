// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/ir/opcodes.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/reg.h"
#include "shader_recompiler/pp_upstream_sample_tee.h"

namespace Shader {

u32 ClassifyImageResourceOperationUse(IR::Opcode opcode,
                                      const IR::TextureInstInfo& inst_info) noexcept {
    using Use = ImageResourceOperationUse;
    const auto mask = [](Use use) { return static_cast<u32>(use); };

    switch (opcode) {
    case IR::Opcode::ImageSampleRaw: {
        u32 result = mask(bool(inst_info.is_depth) ? Use::DepthCompare : Use::RegularSample);
        if (inst_info.is_gather) {
            result |= mask(Use::Gather);
        }
        return result;
    }
    case IR::Opcode::ImageRead:
        return mask(Use::Fetch);
    case IR::Opcode::ImageQueryDimensions:
    case IR::Opcode::ImageQueryLod:
        return mask(Use::Query);
    case IR::Opcode::ImageWrite:
    case IR::Opcode::ImageAtomicIAdd32:
    case IR::Opcode::ImageAtomicSMin32:
    case IR::Opcode::ImageAtomicUMin32:
    case IR::Opcode::ImageAtomicSMax32:
    case IR::Opcode::ImageAtomicUMax32:
    case IR::Opcode::ImageAtomicFMax32:
    case IR::Opcode::ImageAtomicFMin32:
    case IR::Opcode::ImageAtomicInc32:
    case IR::Opcode::ImageAtomicDec32:
    case IR::Opcode::ImageAtomicAnd32:
    case IR::Opcode::ImageAtomicOr32:
    case IR::Opcode::ImageAtomicXor32:
    case IR::Opcode::ImageAtomicExchange32:
    case IR::Opcode::ImageAtomicCmpSwap32:
        return mask(Use::Storage);
    default:
        return 0;
    }
}

bool ImageResourcesShareBindingKey(const ImageResource& lhs,
                                   const ImageResource& rhs) noexcept {
    return lhs.sharp_idx == rhs.sharp_idx && lhs.is_array == rhs.is_array &&
           lhs.mip_fallback_mode == rhs.mip_fallback_mode &&
           lhs.constant_mip_index == rhs.constant_mip_index;
}

void ResetImageResourceOperationUses(Info& info) noexcept {
    info.pp_upstream_image_operation_use_masks.fill(0);
}

bool MergeImageResourceOperationUse(Info& info, u32 binding_index,
                                    u32 operation_use_mask) noexcept {
    if (binding_index >= info.pp_upstream_image_operation_use_masks.size()) {
        return false;
    }
    info.pp_upstream_image_operation_use_masks[binding_index] |= operation_use_mask;
    return true;
}

bool RecordImageResourceOperationUse(Info& info, u32 binding_index, IR::Opcode opcode,
                                     const IR::TextureInstInfo& inst_info) noexcept {
    const u32 operation_use_mask = ClassifyImageResourceOperationUse(opcode, inst_info);
    return operation_use_mask != 0 &&
           MergeImageResourceOperationUse(info, binding_index, operation_use_mask);
}

void ResetSamplerImagePairUses(Info& info) noexcept {
    for (auto& operation_use_masks : info.pp_upstream_sampler_image_operation_use_masks) {
        operation_use_masks.fill(0);
    }
    info.pp_upstream_sampler_image_association_masks.fill(0);
    info.pp_upstream_sampler_image_disable_aniso_masks.fill(0);
    info.pp_upstream_sampler_image_keep_aniso_masks.fill(0);
}

bool MergeSamplerImagePairUse(Info& info, u32 sampler_binding_index,
                              u32 image_binding_index, u32 operation_use_mask,
                              bool disable_aniso) noexcept {
    if (sampler_binding_index >= NUM_SAMPLERS || image_binding_index >= NUM_IMAGES ||
        operation_use_mask == 0) {
        return false;
    }

    info.pp_upstream_sampler_image_operation_use_masks[sampler_binding_index]
                                                           [image_binding_index] |=
        operation_use_mask;
    const u64 image_bit = u64{1} << image_binding_index;
    info.pp_upstream_sampler_image_association_masks[sampler_binding_index] |= image_bit;
    if (disable_aniso) {
        info.pp_upstream_sampler_image_disable_aniso_masks[sampler_binding_index] |= image_bit;
    } else {
        info.pp_upstream_sampler_image_keep_aniso_masks[sampler_binding_index] |= image_bit;
    }
    return true;
}

} // namespace Shader

namespace Shader::Optimization {

bool RecordPpUpstreamImageOperationUseAtBinding(
    Info& info, u32 binding_index, IR::Opcode opcode,
    const IR::TextureInstInfo& inst_info) noexcept {
    return RecordImageResourceOperationUse(info, binding_index, opcode, inst_info);
}

bool RecordPpUpstreamSamplerImagePairAtBindings(
    Info& info, u32 sampler_binding_index, u32 image_binding_index,
    IR::Opcode opcode, const IR::TextureInstInfo& inst_info,
    bool disable_aniso) noexcept {
    if (opcode != IR::Opcode::ImageSampleRaw) {
        return false;
    }
    return MergeSamplerImagePairUse(info, sampler_binding_index, image_binding_index,
                                    ClassifyImageResourceOperationUse(opcode, inst_info),
                                    disable_aniso);
}

} // namespace Shader::Optimization
