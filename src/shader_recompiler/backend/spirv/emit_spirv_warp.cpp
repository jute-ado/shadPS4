// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {
namespace {

Id SubgroupScope(EmitContext& ctx) {
    return ctx.ConstU32(static_cast<u32>(spv::Scope::Subgroup));
}

Id WorkgroupScope(EmitContext& ctx) {
    return ctx.ConstU32(static_cast<u32>(spv::Scope::Workgroup));
}

Id AcquireSemantics(EmitContext& ctx) {
    constexpr auto semantics =
        spv::MemorySemanticsMask::Acquire | spv::MemorySemanticsMask::WorkgroupMemory;
    return ctx.ConstU32(static_cast<u32>(semantics));
}

Id ReleaseSemantics(EmitContext& ctx) {
    constexpr auto semantics =
        spv::MemorySemanticsMask::Release | spv::MemorySemanticsMask::WorkgroupMemory;
    return ctx.ConstU32(static_cast<u32>(semantics));
}

Id SharedPointer(EmitContext& ctx, Id scratch_base, u32 byte_offset) {
    const Id offset =
        byte_offset == 0
            ? scratch_base
            : ctx.OpIAdd(ctx.U32[1], scratch_base, ctx.ConstU32(byte_offset));
    const Id index =
        ctx.OpShiftRightLogical(ctx.U32[1], offset, ctx.ConstU32(2U));
    return ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index);
}

Id SelectSharedPointer(EmitContext& ctx, Id scratch_base, Id is_low, u32 low_byte_offset,
                       u32 high_byte_offset) {
    const Id byte_offset =
        ctx.OpSelect(ctx.U32[1], is_low, ctx.ConstU32(low_byte_offset),
                     ctx.ConstU32(high_byte_offset));
    const Id offset = ctx.OpIAdd(ctx.U32[1], scratch_base, byte_offset);
    const Id index =
        ctx.OpShiftRightLogical(ctx.U32[1], offset, ctx.ConstU32(2U));
    return ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index);
}

Id AtomicLoad(EmitContext& ctx, Id pointer) {
    return ctx.OpAtomicLoad(ctx.U32[1], pointer, WorkgroupScope(ctx), AcquireSemantics(ctx));
}

void AtomicStore(EmitContext& ctx, Id pointer, Id value) {
    // Atomic exchange is used as a release store because Sirit's OpAtomicStore encoder currently
    // emits a result id even though that SPIR-V instruction has no result.
    ctx.OpAtomicExchange(ctx.U32[1], pointer, WorkgroupScope(ctx), ReleaseSemantics(ctx), value);
}

Id IsLowSubgroup(EmitContext& ctx, Id invocation) {
    return ctx.OpULessThan(ctx.U1[1], invocation, ctx.ConstU32(32U));
}

Id SubgroupAny(EmitContext& ctx, Id condition) {
    const Id ballot =
        ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), condition);
    const Id low_mask = ctx.OpCompositeExtract(ctx.U32[1], ballot, 0U);
    return ctx.OpINotEqual(ctx.U1[1], low_mask, ctx.u32_zero_value);
}

Id IsFirstActiveInvocation(EmitContext& ctx, Id invocation) {
    const Id first_invocation =
        ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), invocation);
    return ctx.OpIEqual(ctx.U1[1], invocation, first_invocation);
}

void SpinUntil(EmitContext& ctx, Id pointer, u32 expected) {
    const Id header_label = ctx.OpLabel();
    const Id body_label = ctx.OpLabel();
    const Id continue_label = ctx.OpLabel();
    const Id merge_label = ctx.OpLabel();

    ctx.OpBranch(header_label);
    ctx.AddLabel(header_label);
    ctx.OpLoopMerge(merge_label, continue_label, spv::LoopControlMask::MaskNone);
    ctx.OpBranch(body_label);

    ctx.AddLabel(body_label);
    const Id value = AtomicLoad(ctx, pointer);
    const Id ready = ctx.OpIEqual(ctx.U1[1], value, ctx.ConstU32(expected));
    ctx.OpBranchConditional(ready, merge_label, continue_label);

    ctx.AddLabel(continue_label);
    ctx.OpBranch(header_label);
    ctx.AddLabel(merge_label);
}

constexpr u32 LowActiveOffset = 0;
constexpr u32 HighActiveOffset = 4;
constexpr u32 LowArrivedOffset = 8;
constexpr u32 HighArrivedOffset = 12;
constexpr u32 CandidateOffset = 16;
constexpr u32 CandidateReadyOffset = 20;

} // Anonymous namespace

Id EmitWarpId(EmitContext& ctx) {
    UNREACHABLE();
}

Id EmitLaneId(EmitContext& ctx) {
    return ctx.OpLoad(ctx.U32[1], ctx.subgroup_local_invocation_id);
}

Id EmitQuadShuffle(EmitContext& ctx, Id value, Id index) {
    return ctx.OpGroupNonUniformQuadBroadcast(ctx.U32[1], SubgroupScope(ctx), value, index);
}

Id EmitReadFirstLane(EmitContext& ctx, Id value) {
    return ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), value);
}

Id EmitReadLane(EmitContext& ctx, Id value, Id lane) {
    // ReadLane also represents DS swizzles, whose source lane varies per invocation. SPIR-V
    // broadcast makes a non-uniform invocation id undefined; shuffle-xor permits the permutation.
    const Id invocation_id = ctx.OpLoad(ctx.U32[1], ctx.subgroup_local_invocation_id);
    const Id xor_mask = ctx.OpBitwiseXor(ctx.U32[1], invocation_id, lane);
    return ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value, xor_mask);
}

Id EmitWriteLane(EmitContext& ctx, Id value, Id write_value, u32 lane) {
    return ctx.u32_zero_value;
}

Id EmitBallot(EmitContext& ctx, Id bit) {
    return ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), bit);
}

Id EmitBallotFindLsb(EmitContext& ctx, Id mask) {
    return ctx.OpGroupNonUniformBallotFindLSB(ctx.U32[1], SubgroupScope(ctx), mask);
}

Id EmitGroupAny(EmitContext& ctx, Id bit) {
    return ctx.OpGroupNonUniformAny(ctx.U1[1], SubgroupScope(ctx), bit);
}

void EmitWave64LaneRetirementInit(EmitContext& ctx, Id active, Id invocation, Id scratch_base) {
    const Id is_low = IsLowSubgroup(ctx, invocation);
    const Id active_pointer = SelectSharedPointer(ctx, scratch_base, is_low, LowActiveOffset,
                                                  HighActiveOffset);
    const Id subgroup_active = SubgroupAny(ctx, active);
    const Id active_u32 =
        ctx.OpSelect(ctx.U32[1], subgroup_active, ctx.ConstU32(1U), ctx.u32_zero_value);
    AtomicStore(ctx, active_pointer, active_u32);
}

void EmitWave64LaneRetirementSync(EmitContext& ctx, Id continues, Id invocation, Id scratch_base) {
    const Id is_low = IsLowSubgroup(ctx, invocation);
    const Id is_first = IsFirstActiveInvocation(ctx, invocation);
    const Id subgroup_continues = SubgroupAny(ctx, continues);
    const Id continues_u32 =
        ctx.OpSelect(ctx.U32[1], subgroup_continues, ctx.ConstU32(1U), ctx.u32_zero_value);

    const Id first_label = ctx.OpLabel();
    const Id first_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(first_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_first, first_label, first_merge_label);

    ctx.AddLabel(first_label);
    const Id self_active = SelectSharedPointer(ctx, scratch_base, is_low, LowActiveOffset,
                                               HighActiveOffset);
    const Id peer_active = SelectSharedPointer(ctx, scratch_base, is_low, HighActiveOffset,
                                               LowActiveOffset);
    const Id self_arrived = SelectSharedPointer(ctx, scratch_base, is_low, LowArrivedOffset,
                                                HighArrivedOffset);
    const Id peer_arrived = SelectSharedPointer(ctx, scratch_base, is_low, HighArrivedOffset,
                                                LowArrivedOffset);
    const Id peer_was_active_u32 = AtomicLoad(ctx, peer_active);
    const Id peer_was_active =
        ctx.OpINotEqual(ctx.U1[1], peer_was_active_u32, ctx.u32_zero_value);

    // Both subgroups must read the previous active state before either publishes retirement.
    // The asymmetric arrival protocol prevents a fast subgroup from resetting its flag before
    // the peer has observed it.
    const Id phase_one_label = ctx.OpLabel();
    const Id phase_one_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(phase_one_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(peer_was_active, phase_one_label, phase_one_merge_label);

    ctx.AddLabel(phase_one_label);
    const Id low_phase_one_label = ctx.OpLabel();
    const Id high_phase_one_label = ctx.OpLabel();
    const Id side_phase_one_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(side_phase_one_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_low, low_phase_one_label, high_phase_one_label);

    ctx.AddLabel(low_phase_one_label);
    AtomicStore(ctx, self_arrived, ctx.ConstU32(1U));
    SpinUntil(ctx, peer_arrived, 1U);
    ctx.OpBranch(side_phase_one_merge_label);

    ctx.AddLabel(high_phase_one_label);
    SpinUntil(ctx, peer_arrived, 1U);
    AtomicStore(ctx, self_arrived, ctx.ConstU32(1U));
    ctx.OpBranch(side_phase_one_merge_label);

    ctx.AddLabel(side_phase_one_merge_label);
    ctx.OpBranch(phase_one_merge_label);
    ctx.AddLabel(phase_one_merge_label);

    AtomicStore(ctx, self_active, continues_u32);

    // Complete the second half of the rendezvous only after the new active state is visible.
    const Id phase_two_label = ctx.OpLabel();
    const Id phase_two_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(phase_two_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(peer_was_active, phase_two_label, phase_two_merge_label);

    ctx.AddLabel(phase_two_label);
    const Id low_phase_two_label = ctx.OpLabel();
    const Id high_phase_two_label = ctx.OpLabel();
    const Id side_phase_two_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(side_phase_two_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_low, low_phase_two_label, high_phase_two_label);

    ctx.AddLabel(low_phase_two_label);
    AtomicStore(ctx, self_arrived, ctx.u32_zero_value);
    SpinUntil(ctx, peer_arrived, 0U);
    ctx.OpBranch(side_phase_two_merge_label);

    ctx.AddLabel(high_phase_two_label);
    SpinUntil(ctx, peer_arrived, 0U);
    AtomicStore(ctx, self_arrived, ctx.u32_zero_value);
    ctx.OpBranch(side_phase_two_merge_label);

    ctx.AddLabel(side_phase_two_merge_label);
    ctx.OpBranch(phase_two_merge_label);
    ctx.AddLabel(phase_two_merge_label);
    ctx.OpBranch(first_merge_label);

    ctx.AddLabel(first_merge_label);
    ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), invocation);
}

Id EmitWave64ReadFirstLane(EmitContext& ctx, Id value, Id invocation, Id scratch_base) {
    const Id native_result =
        ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), value);
    const Id is_low = IsLowSubgroup(ctx, invocation);
    const Id is_first = IsFirstActiveInvocation(ctx, invocation);
    const Id peer_active_pointer =
        SelectSharedPointer(ctx, scratch_base, is_low, HighActiveOffset, LowActiveOffset);
    const Id peer_active_u32 = AtomicLoad(ctx, peer_active_pointer);
    const Id peer_active =
        ctx.OpINotEqual(ctx.U1[1], peer_active_u32, ctx.u32_zero_value);

    const Id no_peer_label = ctx.last_label;
    const Id exchange_label = ctx.OpLabel();
    const Id final_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(final_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(peer_active, exchange_label, final_merge_label);

    ctx.AddLabel(exchange_label);
    const Id low_label = ctx.OpLabel();
    const Id high_label = ctx.OpLabel();
    const Id exchange_merge_label = ctx.OpLabel();
    const Id candidate_pointer = SharedPointer(ctx, scratch_base, CandidateOffset);
    const Id ready_pointer = SharedPointer(ctx, scratch_base, CandidateReadyOffset);
    ctx.OpSelectionMerge(exchange_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_low, low_label, high_label);

    // When both subgroups remain active, the low subgroup owns the first active guest lane. Its
    // first active invocation publishes the native subgroup result, and the high subgroup
    // acknowledges consumption before the next loop iteration may overwrite it.
    ctx.AddLabel(low_label);
    SpinUntil(ctx, ready_pointer, 0U);
    const Id low_first_label = ctx.OpLabel();
    const Id low_first_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(low_first_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_first, low_first_label, low_first_merge_label);
    ctx.AddLabel(low_first_label);
    AtomicStore(ctx, candidate_pointer, native_result);
    AtomicStore(ctx, ready_pointer, ctx.ConstU32(1U));
    ctx.OpBranch(low_first_merge_label);
    ctx.AddLabel(low_first_merge_label);
    ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), native_result);
    SpinUntil(ctx, ready_pointer, 0U);
    const Id low_exit_label = ctx.last_label;
    ctx.OpBranch(exchange_merge_label);

    ctx.AddLabel(high_label);
    SpinUntil(ctx, ready_pointer, 1U);
    const Id candidate = AtomicLoad(ctx, candidate_pointer);
    const Id high_result =
        ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), candidate);
    const Id high_first_label = ctx.OpLabel();
    const Id high_first_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(high_first_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_first, high_first_label, high_first_merge_label);
    ctx.AddLabel(high_first_label);
    AtomicStore(ctx, ready_pointer, ctx.u32_zero_value);
    ctx.OpBranch(high_first_merge_label);
    ctx.AddLabel(high_first_merge_label);
    ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), high_result);
    const Id high_exit_label = ctx.last_label;
    ctx.OpBranch(exchange_merge_label);

    ctx.AddLabel(exchange_merge_label);
    const Id exchange_result =
        ctx.OpPhi(ctx.U32[1], native_result, low_exit_label, high_result, high_exit_label);
    ctx.OpBranch(final_merge_label);

    ctx.AddLabel(final_merge_label);
    return ctx.OpPhi(ctx.U32[1], native_result, no_peer_label, exchange_result,
                     exchange_merge_label);
}

} // namespace Shader::Backend::SPIRV
