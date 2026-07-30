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

void SpinUntilAtLeast(EmitContext& ctx, Id pointer, Id expected) {
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
    const Id ready = ctx.OpUGreaterThanEqual(ctx.U1[1], value, expected);
    ctx.OpBranchConditional(ready, merge_label, continue_label);

    ctx.AddLabel(continue_label);
    ctx.OpBranch(header_label);
    ctx.AddLabel(merge_label);
}

constexpr u32 LowActiveOffset = 0;
constexpr u32 HighActiveOffset = 4;
constexpr u32 LowGenerationOffset = 8;
constexpr u32 HighGenerationOffset = 12;
constexpr u32 CandidateOffset = 16;
constexpr u32 CandidateGenerationOffset = 20;

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
    const Id is_first = IsFirstActiveInvocation(ctx, invocation);
    const Id active_pointer = SelectSharedPointer(ctx, scratch_base, is_low, LowActiveOffset,
                                                  HighActiveOffset);
    const Id generation_pointer =
        SelectSharedPointer(ctx, scratch_base, is_low, LowGenerationOffset, HighGenerationOffset);
    const Id candidate_generation_pointer =
        SharedPointer(ctx, scratch_base, CandidateGenerationOffset);
    const Id subgroup_active = SubgroupAny(ctx, active);
    const Id active_u32 =
        ctx.OpSelect(ctx.U32[1], subgroup_active, ctx.ConstU32(1U), ctx.u32_zero_value);

    const Id first_label = ctx.OpLabel();
    const Id first_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(first_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_first, first_label, first_merge_label);
    ctx.AddLabel(first_label);
    AtomicStore(ctx, active_pointer, active_u32);
    AtomicStore(ctx, generation_pointer, ctx.u32_zero_value);
    AtomicStore(ctx, candidate_generation_pointer, ctx.u32_zero_value);
    ctx.OpBranch(first_merge_label);
    ctx.AddLabel(first_merge_label);
    ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), invocation);
}

void EmitWave64LaneRetirementSync(EmitContext& ctx, Id continues, Id invocation, Id scratch_base) {
    const Id is_low = IsLowSubgroup(ctx, invocation);
    const Id is_first = IsFirstActiveInvocation(ctx, invocation);
    const Id subgroup_continues = SubgroupAny(ctx, continues);
    const Id continues_u32 =
        ctx.OpSelect(ctx.U32[1], subgroup_continues, ctx.ConstU32(1U), ctx.u32_zero_value);
    const Id self_active = SelectSharedPointer(ctx, scratch_base, is_low, LowActiveOffset,
                                               HighActiveOffset);
    const Id peer_active = SelectSharedPointer(ctx, scratch_base, is_low, HighActiveOffset,
                                               LowActiveOffset);
    const Id self_generation =
        SelectSharedPointer(ctx, scratch_base, is_low, LowGenerationOffset, HighGenerationOffset);
    const Id peer_generation =
        SelectSharedPointer(ctx, scratch_base, is_low, HighGenerationOffset, LowGenerationOffset);

    const Id first_label = ctx.OpLabel();
    const Id first_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(first_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_first, first_label, first_merge_label);

    ctx.AddLabel(first_label);
    const Id previous_generation = AtomicLoad(ctx, self_generation);
    const Id next_generation =
        ctx.OpIAdd(ctx.U32[1], previous_generation, ctx.ConstU32(1U));
    AtomicStore(ctx, self_active, continues_u32);
    // Publish the generation last. An acquire load that observes it also observes the subgroup's
    // retirement state, and neither subgroup can miss a one-way retirement transition.
    AtomicStore(ctx, self_generation, next_generation);
    ctx.OpBranch(first_merge_label);

    ctx.AddLabel(first_merge_label);
    ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), invocation);

    const Id generation = AtomicLoad(ctx, self_generation);
    const Id peer_was_active_u32 = AtomicLoad(ctx, peer_active);
    const Id peer_was_active =
        ctx.OpINotEqual(ctx.U1[1], peer_was_active_u32, ctx.u32_zero_value);
    const Id wait_label = ctx.OpLabel();
    const Id wait_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(wait_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(peer_was_active, wait_label, wait_merge_label);
    ctx.AddLabel(wait_label);
    SpinUntilAtLeast(ctx, peer_generation, generation);
    ctx.OpBranch(wait_merge_label);
    ctx.AddLabel(wait_merge_label);
}

Id EmitWave64ReadFirstLane(EmitContext& ctx, Id value, Id invocation, Id scratch_base) {
    const Id native_result =
        ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), value);
    const Id is_low = IsLowSubgroup(ctx, invocation);
    const Id is_first = IsFirstActiveInvocation(ctx, invocation);
    const Id self_generation =
        SelectSharedPointer(ctx, scratch_base, is_low, LowGenerationOffset, HighGenerationOffset);
    const Id peer_active_pointer =
        SelectSharedPointer(ctx, scratch_base, is_low, HighActiveOffset, LowActiveOffset);
    const Id generation = AtomicLoad(ctx, self_generation);
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
    const Id candidate_generation_pointer =
        SharedPointer(ctx, scratch_base, CandidateGenerationOffset);
    ctx.OpSelectionMerge(exchange_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_low, low_label, high_label);

    // When both subgroups remain active, the low subgroup owns the first active guest lane. Its
    // first active invocation publishes the native subgroup result followed by its generation.
    // The next retirement synchronization prevents it from overwriting the candidate before the
    // high subgroup has consumed this generation.
    ctx.AddLabel(low_label);
    const Id low_first_label = ctx.OpLabel();
    const Id low_first_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(low_first_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_first, low_first_label, low_first_merge_label);
    ctx.AddLabel(low_first_label);
    AtomicStore(ctx, candidate_pointer, native_result);
    AtomicStore(ctx, candidate_generation_pointer, generation);
    ctx.OpBranch(low_first_merge_label);
    ctx.AddLabel(low_first_merge_label);
    ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), native_result);
    const Id low_exit_label = ctx.last_label;
    ctx.OpBranch(exchange_merge_label);

    ctx.AddLabel(high_label);
    SpinUntilAtLeast(ctx, candidate_generation_pointer, generation);
    const Id candidate = AtomicLoad(ctx, candidate_pointer);
    const Id high_result =
        ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), candidate);
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
