// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/ir/debug_print.h"
#include "shader_recompiler/pp_upstream_sample_tee.h"

namespace Shader::Backend::SPIRV {

void EmitPpUpstreamSampleTee(EmitContext& ctx, Id value, u32 resource_index,
                             Id cube_metadata, Id dependency_metadata) {
    if (ctx.pp_upstream_sample_tee_buffer_index == EmitContext::InvalidBufferIndex) {
        return;
    }
    ASSERT(ctx.stage == Stage::Fragment && Sirit::ValidId(ctx.frag_coord));
    ASSERT(resource_index < NUM_IMAGES);

    const Id push_u32_ptr{ctx.TypePointer(spv::StorageClass::PushConstant, ctx.U32[1])};
    const auto load_push_component = [&](u32 component) {
        const Id pointer{ctx.OpAccessChain(push_u32_ptr, ctx.push_data_block,
                                           ctx.ConstU32(PushData::PpUpstreamSampleTeeMemberIndex),
                                           ctx.ConstU32(component))};
        return ctx.OpLoad(ctx.U32[1], pointer);
    };
    const Id control{load_push_component(PushData::PpUpstreamSampleTeeControlComponent)};
    const Id base_dword{load_push_component(PushData::PpUpstreamSampleTeeBaseDwordComponent)};

    const Id armed{ctx.OpIEqual(
        ctx.U1[1], ctx.OpBitwiseAnd(ctx.U32[1], control, ctx.ConstU32(PpUpstreamSampleTeeArmMask)),
        ctx.ConstU32(PpUpstreamSampleTeeArmMask))};
    const Id resource_matches{ctx.OpIEqual(
        ctx.U1[1],
        ctx.OpBitwiseAnd(ctx.U32[1], control, ctx.ConstU32(PpUpstreamSampleTeeResourceMask)),
        ctx.ConstU32(resource_index))};
    const Id reserved_clear{ctx.OpIEqual(
        ctx.U1[1],
        ctx.OpBitwiseAnd(ctx.U32[1], control, ctx.ConstU32(PpUpstreamSampleTeeReservedMask)),
        ctx.u32_zero_value)};
    const Id base_aligned{ctx.OpIEqual(
        ctx.U1[1],
        ctx.OpUMod(ctx.U32[1], base_dword, ctx.ConstU32(PpUpstreamSampleTeeSlotStrideDwords)),
        ctx.u32_zero_value)};
    const Id base_in_range{ctx.OpULessThanEqual(
        ctx.U1[1], base_dword,
        ctx.ConstU32(PpUpstreamSampleTeeBufferDwords - PpUpstreamSampleTeeSlotStrideDwords))};
    const Id control_valid{ctx.OpLogicalAnd(
        ctx.U1[1], armed,
        ctx.OpLogicalAnd(
            ctx.U1[1], resource_matches,
            ctx.OpLogicalAnd(ctx.U1[1], reserved_clear,
                             ctx.OpLogicalAnd(ctx.U1[1], base_aligned, base_in_range))))};
    const Id candidate{ctx.OpBitFieldUExtract(
        ctx.U32[1], control, ctx.ConstU32(PpUpstreamSampleTeeCandidateShift), ctx.ConstU32(2u))};

    const Id header_label{ctx.OpLabel()};
    const Id merge_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(control_valid, header_label, merge_label);
    ctx.AddLabel(header_label);

    const auto& buffer = ctx.buffers[ctx.pp_upstream_sample_tee_buffer_index];
    const auto [buffer_id, pointer_type] = buffer.Alias(EmitContext::PointerType::U32);
    const auto pointer_at = [&](Id dword) {
        return ctx.OpAccessChain(pointer_type, buffer_id, ctx.u32_zero_value, dword);
    };
    const auto load_header = [&](u32 offset) {
        return ctx.OpLoad(ctx.U32[1],
                          pointer_at(ctx.OpIAdd(ctx.U32[1], base_dword, ctx.ConstU32(offset))));
    };
    const Id magic{load_header(0)};
    const Id generation{load_header(1)};
    const Id header_candidate{load_header(2)};
    const Id origin_x{load_header(3)};
    const Id origin_y{load_header(4)};
    const Id width{load_header(5)};
    const Id height{load_header(6)};
    const Id pixel_count{load_header(7)};

    Id header_valid{ctx.OpIEqual(ctx.U1[1], magic, ctx.ConstU32(PpUpstreamSampleTeeHeaderMagic))};
    const auto append_header_condition = [&](Id condition) {
        header_valid = ctx.OpLogicalAnd(ctx.U1[1], header_valid, condition);
    };
    append_header_condition(ctx.OpINotEqual(ctx.U1[1], generation, ctx.u32_zero_value));
    append_header_condition(ctx.OpIEqual(ctx.U1[1], header_candidate, candidate));
    append_header_condition(ctx.OpUGreaterThanEqual(ctx.U1[1], width, ctx.u32_one_value));
    append_header_condition(
        ctx.OpULessThanEqual(ctx.U1[1], width, ctx.ConstU32(PpUpstreamSampleTeeMaxDimension)));
    append_header_condition(ctx.OpUGreaterThanEqual(ctx.U1[1], height, ctx.u32_one_value));
    append_header_condition(
        ctx.OpULessThanEqual(ctx.U1[1], height, ctx.ConstU32(PpUpstreamSampleTeeMaxDimension)));
    append_header_condition(
        ctx.OpIEqual(ctx.U1[1], pixel_count, ctx.OpIMul(ctx.U32[1], width, height)));
    const Id required_dwords{ctx.OpIAdd(
        ctx.U32[1], ctx.ConstU32(PpUpstreamSampleTeeHeaderDwords),
        ctx.OpIMul(ctx.U32[1], pixel_count, ctx.ConstU32(PpUpstreamSampleTeeRecordDwords)))};
    append_header_condition(ctx.OpULessThanEqual(
        ctx.U1[1], required_dwords, ctx.ConstU32(PpUpstreamSampleTeeSlotStrideDwords)));
    append_header_condition(
        ctx.OpULessThanEqual(ctx.U1[1], ctx.OpIAdd(ctx.U32[1], base_dword, required_dwords),
                             ctx.ConstU32(PpUpstreamSampleTeeBufferDwords)));

    const Id write_label{ctx.OpLabel()};
    const Id header_merge_label{ctx.OpLabel()};
    const Id frag_coord{ctx.OpLoad(ctx.F32[4], ctx.frag_coord)};
    const Id pixel_x{
        ctx.OpConvertFToU(ctx.U32[1], ctx.OpCompositeExtract(ctx.F32[1], frag_coord, 0))};
    const Id pixel_y{
        ctx.OpConvertFToU(ctx.U32[1], ctx.OpCompositeExtract(ctx.F32[1], frag_coord, 1))};
    const Id delta_x{ctx.OpISub(ctx.U32[1], pixel_x, origin_x)};
    const Id delta_y{ctx.OpISub(ctx.U32[1], pixel_y, origin_y)};
    const Id x_in_range{ctx.OpLogicalAnd(ctx.U1[1],
                                         ctx.OpUGreaterThanEqual(ctx.U1[1], pixel_x, origin_x),
                                         ctx.OpULessThan(ctx.U1[1], delta_x, width))};
    const Id y_in_range{ctx.OpLogicalAnd(ctx.U1[1],
                                         ctx.OpUGreaterThanEqual(ctx.U1[1], pixel_y, origin_y),
                                         ctx.OpULessThan(ctx.U1[1], delta_y, height))};
    const Id write_valid{ctx.OpLogicalAnd(ctx.U1[1], header_valid,
                                          ctx.OpLogicalAnd(ctx.U1[1], x_in_range, y_in_range))};
    ctx.OpSelectionMerge(header_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(write_valid, write_label, header_merge_label);
    ctx.AddLabel(write_label);

    const Id local_index{ctx.OpIAdd(ctx.U32[1], ctx.OpIMul(ctx.U32[1], delta_y, width), delta_x)};
    const Id record_dword{ctx.OpIAdd(
        ctx.U32[1],
        ctx.OpIAdd(ctx.U32[1], base_dword, ctx.ConstU32(PpUpstreamSampleTeeHeaderDwords)),
        ctx.OpIMul(ctx.U32[1], local_index, ctx.ConstU32(PpUpstreamSampleTeeRecordDwords)))};
    const Id writer_dword{ctx.OpIAdd(ctx.U32[1], record_dword, ctx.ConstU32(6u))};
    const Id previous_writers{ctx.OpAtomicIAdd(ctx.U32[1], pointer_at(writer_dword),
                                               ctx.ConstU32(static_cast<u32>(spv::Scope::Device)),
                                               ctx.u32_zero_value, ctx.u32_one_value)};
    const Id is_winner{ctx.OpIEqual(ctx.U1[1], previous_writers, ctx.u32_zero_value)};
    const Id winner_label{ctx.OpLabel()};
    const Id writer_merge_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(writer_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_winner, winner_label, writer_merge_label);
    ctx.AddLabel(winner_label);
    const Id raw_value{ctx.OpBitcast(ctx.U32[4], value)};
    for (u32 component = 0; component < 4; ++component) {
        const Id component_dword{ctx.OpIAdd(ctx.U32[1], record_dword, ctx.ConstU32(component))};
        ctx.OpStore(pointer_at(component_dword),
                    ctx.OpCompositeExtract(ctx.U32[1], raw_value, component));
    }
    const Id cube_metadata_dword{ctx.OpIAdd(ctx.U32[1], record_dword, ctx.ConstU32(4u))};
    ctx.OpStore(pointer_at(cube_metadata_dword), cube_metadata);
    const Id dependency_metadata_dword{
        ctx.OpIAdd(ctx.U32[1], record_dword, ctx.ConstU32(5u))};
    ctx.OpStore(pointer_at(dependency_metadata_dword), dependency_metadata);
    ctx.OpBranch(writer_merge_label);

    ctx.AddLabel(writer_merge_label);
    ctx.OpBranch(header_merge_label);

    ctx.AddLabel(header_merge_label);
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

void EmitPrologue(EmitContext& ctx) {
    if (ctx.stage == Stage::Fragment) {
        ctx.DefineAmdPerVertexAttribs();
    }
    if (ctx.info.loads.Get(IR::Attribute::WorkgroupIndex)) {
        ctx.DefineWorkgroupIndex();
    }
    ctx.DefineBufferProperties();
}

void ConvertDepthMode(EmitContext& ctx) {
    const Id type{ctx.F32[1]};
    const Id position{ctx.OpLoad(ctx.F32[4], ctx.output_position)};
    const Id z{ctx.OpCompositeExtract(type, position, 2u)};
    const Id w{ctx.OpCompositeExtract(type, position, 3u)};
    const Id screen_depth{ctx.OpFMul(type, ctx.OpFAdd(type, z, w), ctx.Constant(type, 0.5f))};
    const Id vector{ctx.OpCompositeInsert(ctx.F32[4], screen_depth, position, 2u)};
    ctx.OpStore(ctx.output_position, vector);
}

void ConvertPositionToClipSpace(EmitContext& ctx) {
    ASSERT_MSG(!ctx.info.stores.GetAny(IR::Attribute::ViewportIndex),
               "Multi-viewport with shader clip space conversion not yet implemented.");

    const Id type{ctx.F32[1]};
    Id position{ctx.OpLoad(ctx.F32[4], ctx.output_position)};
    const Id x{ctx.OpCompositeExtract(type, position, 0u)};
    const Id y{ctx.OpCompositeExtract(type, position, 1u)};
    const Id z{ctx.OpCompositeExtract(type, position, 2u)};
    const Id w{ctx.OpCompositeExtract(type, position, 3u)};
    const Id xoffset_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                           ctx.push_data_block,
                                           ctx.ConstU32(PushData::XOffsetIndex))};
    const Id xoffset{ctx.OpLoad(type, xoffset_ptr)};
    const Id yoffset_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                           ctx.push_data_block,
                                           ctx.ConstU32(PushData::YOffsetIndex))};
    const Id yoffset{ctx.OpLoad(type, yoffset_ptr)};
    const Id xscale_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                          ctx.push_data_block,
                                          ctx.ConstU32(PushData::XScaleIndex))};
    const Id xscale{ctx.OpLoad(type, xscale_ptr)};
    const Id yscale_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                          ctx.push_data_block,
                                          ctx.ConstU32(PushData::YScaleIndex))};
    const Id yscale{ctx.OpLoad(type, yscale_ptr)};
    const Id vport_w =
        ctx.Constant(type, float(std::min<u32>(ctx.profile.max_viewport_width / 2, 8_KB)));
    const Id wnd_x = ctx.OpFAdd(type, ctx.OpFMul(type, x, xscale), xoffset);
    const Id ndc_x = ctx.OpFSub(type, ctx.OpFDiv(type, wnd_x, vport_w), ctx.Constant(type, 1.f));
    const Id vport_h =
        ctx.Constant(type, float(std::min<u32>(ctx.profile.max_viewport_height / 2, 8_KB)));
    const Id wnd_y = ctx.OpFAdd(type, ctx.OpFMul(type, y, yscale), yoffset);
    const Id ndc_y = ctx.OpFSub(type, ctx.OpFDiv(type, wnd_y, vport_h), ctx.Constant(type, 1.f));
    const Id vector{ctx.OpCompositeConstruct(ctx.F32[4], std::array<Id, 4>({ndc_x, ndc_y, z, w}))};
    ctx.OpStore(ctx.output_position, vector);
}

void EmitEpilogue(EmitContext& ctx) {
    if (ctx.stage == Stage::Vertex && ctx.runtime_info.vs_info.emulate_depth_negative_one_to_one) {
        ConvertDepthMode(ctx);
    }
    if (ctx.stage == Stage::Vertex && ctx.runtime_info.vs_info.clip_disable) {
        ConvertPositionToClipSpace(ctx);
    }
}

void EmitDiscard(EmitContext& ctx) {
    ctx.OpDemoteToHelperInvocationEXT();
}

void EmitDiscardCond(EmitContext& ctx, Id condition) {
    const Id kill_label{ctx.OpLabel()};
    const Id merge_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(condition, kill_label, merge_label);
    ctx.AddLabel(kill_label);
    ctx.OpDemoteToHelperInvocationEXT();
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

void EmitEmitVertex(EmitContext& ctx) {
    ctx.OpEmitVertex();
}

void EmitEmitPrimitive(EmitContext& ctx) {
    ctx.OpEndPrimitive();
}

void EmitEmitVertex(EmitContext& ctx, const IR::Value& stream) {
    UNREACHABLE_MSG("Geometry streams");
}

void EmitEndPrimitive(EmitContext& ctx, const IR::Value& stream) {
    UNREACHABLE_MSG("Geometry streams");
}

void EmitDebugPrint(EmitContext& ctx, IR::Inst* inst, Id fmt, Id arg0, Id arg1, Id arg2, Id arg3) {
    IR::DebugPrintFlags flags = inst->Flags<IR::DebugPrintFlags>();
    std::array<Id, IR::DEBUGPRINT_NUM_FORMAT_ARGS> fmt_args = {arg0, arg1, arg2, arg3};
    auto fmt_args_span = std::span<Id>(fmt_args.begin(), fmt_args.begin() + flags.num_args);
    ctx.OpDebugPrintf(fmt, fmt_args_span);
}

} // namespace Shader::Backend::SPIRV
