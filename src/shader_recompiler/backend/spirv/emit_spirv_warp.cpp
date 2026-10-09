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
    const Id offset = byte_offset == 0
                          ? scratch_base
                          : ctx.OpIAdd(ctx.U32[1], scratch_base, ctx.ConstU32(byte_offset));
    const Id index = ctx.OpShiftRightLogical(ctx.U32[1], offset, ctx.ConstU32(2U));
    return ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index);
}

Id SelectSharedPointer(EmitContext& ctx, Id scratch_base, Id is_low, u32 low_byte_offset,
                       u32 high_byte_offset) {
    const Id byte_offset =
        ctx.OpSelect(ctx.U32[1], is_low, ctx.ConstU32(low_byte_offset),
                     ctx.ConstU32(high_byte_offset));
    const Id offset = ctx.OpIAdd(ctx.U32[1], scratch_base, byte_offset);
    const Id index = ctx.OpShiftRightLogical(ctx.U32[1], offset, ctx.ConstU32(2U));
    return ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index);
}

Id AtomicLoad(EmitContext& ctx, Id pointer) {
    return ctx.OpAtomicLoad(ctx.U32[1], pointer, WorkgroupScope(ctx), AcquireSemantics(ctx));
}

void AtomicStore(EmitContext& ctx, Id pointer, Id value) {
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

struct FirstActiveValue {
    Id has_active;
    Id value;
};

FirstActiveValue SelectFirstActiveValue(EmitContext& ctx, Id value, Id active) {
    const Id ballot =
        ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), active);
    Id any_bits = ctx.OpCompositeExtract(ctx.U32[1], ballot, 0U);
    for (u32 word = 1; word < 4; ++word) {
        any_bits = ctx.OpBitwiseOr(ctx.U32[1], any_bits,
                                   ctx.OpCompositeExtract(ctx.U32[1], ballot, word));
    }
    const Id has_active =
        ctx.OpINotEqual(ctx.U1[1], any_bits, ctx.u32_zero_value);
    const Id first_active =
        ctx.OpGroupNonUniformBallotFindLSB(ctx.U32[1], SubgroupScope(ctx), ballot);
    // GCN leaves the result undefined for EXEC=0. Select physical lane zero in that case so the
    // host broadcast still receives an in-range index and remains deterministic.
    const Id source_lane =
        ctx.OpSelect(ctx.U32[1], has_active, first_active, ctx.u32_zero_value);
    return {
        .has_active = has_active,
        .value = ctx.OpGroupNonUniformBroadcast(ctx.U32[1], SubgroupScope(ctx), value,
                                                source_lane),
    };
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
    const Id ready =
        ctx.OpUGreaterThanEqual(ctx.U1[1], AtomicLoad(ctx, pointer), expected);
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

Id EmitReadFirstLane(EmitContext& ctx, Id value, Id active) {
    return SelectFirstActiveValue(ctx, value, active).value;
}

Id EmitReadLane(EmitContext& ctx, Id value, Id lane) {
    return ctx.OpGroupNonUniformBroadcast(ctx.U32[1], SubgroupScope(ctx), value, lane);
}

Id EmitWriteLane(EmitContext& ctx, Id value, Id write_value, u32 lane) {
    return ctx.u32_zero_value;
}

Id EmitBallot(EmitContext& ctx, Id bit) {
    return ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), bit);
}

Id EmitCaptureScalarMaskWord(EmitContext&, Id, Id scalar, u32) {
    // The wave64 lowering replaces captures at proven workgroup-converged producer sites. Until
    // then retain the prior scalar fallback; materialization keeps the established fail-closed
    // behavior for split-wave hosts.
    return scalar;
}

Id EmitShuffleXor(EmitContext& ctx, Id value, Id mask) {
    return ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value, mask);
}

Id EmitMaterializeScalarMaskWord(EmitContext& ctx, Id is_thread_bit, Id bit, Id scalar,
                                 u32 word) {
    const Id ballot_label = ctx.OpLabel();
    const Id scalar_label = ctx.OpLabel();
    const Id merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_thread_bit, ballot_label, scalar_label);

    ctx.AddLabel(ballot_label);
    const Id ballot = EmitBallot(ctx, bit);
    Id ballot_word{};
    if (ctx.info.emulate_wave64) {
        ASSERT(ctx.profile.subgroup_size == 32 && ctx.stage == Stage::Compute);
        const Id guest_wave_word = ctx.OpLoad(ctx.U32[1], ctx.subgroup_id);
        const Id is_matching_half =
            ctx.OpIEqual(ctx.U1[1], guest_wave_word, ctx.ConstU32(word));
        const Id local_ballot_word = ctx.OpCompositeExtract(ctx.U32[1], ballot, 0);
        ballot_word =
            ctx.OpSelect(ctx.U32[1], is_matching_half, local_ballot_word, scalar);
    } else {
        ballot_word = ctx.OpCompositeExtract(ctx.U32[1], ballot, word);
    }
    ctx.OpBranch(merge_label);

    ctx.AddLabel(scalar_label);
    ctx.OpBranch(merge_label);

    ctx.AddLabel(merge_label);
    return ctx.OpPhi(ctx.U32[1], ballot_word, ballot_label, scalar, scalar_label);
}

Id EmitBallotFindLsb(EmitContext& ctx, Id mask) {
    return ctx.OpGroupNonUniformBallotFindLSB(ctx.U32[1], SubgroupScope(ctx), mask);
}

Id EmitGroupAny(EmitContext& ctx, Id bit) {
    return ctx.OpGroupNonUniformAny(ctx.U1[1], SubgroupScope(ctx), bit);
}

Id EmitGuestWaveAny(EmitContext& ctx, Id bit) {
    // A surviving GuestWaveAny means the complete 64-lane lowering declined. Preserve the old
    // per-invocation control on a split-wave host instead of silently changing it to a wave32
    // vote. Native wave64 can implement the guest scalar branch directly.
    return ctx.profile.subgroup_size >= 64 ? EmitGroupAny(ctx, bit) : bit;
}

Id EmitWave64GroupAny(EmitContext& ctx, Id bit, Id invocation, Id scratch_base) {
    const Id native_any = EmitGroupAny(ctx, bit);
    const Id native_u32 =
        ctx.OpSelect(ctx.U32[1], native_any, ctx.ConstU32(1U), ctx.u32_zero_value);
    const Id is_low = IsLowSubgroup(ctx, invocation);
    const Id is_first = IsFirstActiveInvocation(ctx, invocation);
    const Id self_pointer = SelectSharedPointer(ctx, scratch_base, is_low, 0U, sizeof(u32));

    const Id publish_label = ctx.OpLabel();
    const Id publish_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(publish_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_first, publish_label, publish_merge_label);
    ctx.AddLabel(publish_label);
    AtomicStore(ctx, self_pointer, native_u32);
    ctx.OpBranch(publish_merge_label);
    ctx.AddLabel(publish_merge_label);

    // Scalar GCN branches test the complete 64-lane VCC/EXEC mask. This operation is emitted
    // only at a structurally converged scalar block, so both subgroup32 halves can rendezvous.
    EmitBarrier(ctx);
    const Id low = AtomicLoad(ctx, SharedPointer(ctx, scratch_base, 0U));
    const Id high = AtomicLoad(ctx, SharedPointer(ctx, scratch_base, sizeof(u32)));
    const Id result = ctx.OpINotEqual(ctx.U1[1], ctx.OpBitwiseOr(ctx.U32[1], low, high),
                                     ctx.u32_zero_value);
    // The same two words are reused by later scalar branches.
    EmitBarrier(ctx);
    return result;
}

Id EmitWave64CaptureScalarMaskWord(EmitContext& ctx, Id bit, u32 word, Id invocation,
                                   Id scratch_base) {
    ASSERT(word < 2U);
    const Id ballot = EmitBallot(ctx, bit);
    const Id local_word = ctx.OpCompositeExtract(ctx.U32[1], ballot, 0U);
    const Id is_low = IsLowSubgroup(ctx, invocation);
    const Id is_first = IsFirstActiveInvocation(ctx, invocation);
    const Id self_pointer = SelectSharedPointer(ctx, scratch_base, is_low, 0U, sizeof(u32));

    const Id publish_label = ctx.OpLabel();
    const Id publish_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(publish_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_first, publish_label, publish_merge_label);
    ctx.AddLabel(publish_label);
    AtomicStore(ctx, self_pointer, local_word);
    ctx.OpBranch(publish_merge_label);
    ctx.AddLabel(publish_merge_label);

    // Capture sites are scalar GCN instructions in guest-wave-uniform structured regions.
    // Both subgroup32 halves therefore publish before either half consumes the selected word.
    EmitBarrier(ctx);
    const Id guest_word =
        AtomicLoad(ctx, SharedPointer(ctx, scratch_base, word * sizeof(u32)));
    EmitBarrier(ctx);
    return guest_word;
}

Id EmitWave64LockstepMaterializeScalarMaskWord(EmitContext& ctx, Id is_thread_bit, Id bit,
                                               Id scalar, u32 word, Id invocation,
                                               Id scratch_base) {
    ASSERT(word < 2U);
    const Id ballot = EmitBallot(ctx, bit);
    const Id local_word = ctx.OpCompositeExtract(ctx.U32[1], ballot, 0U);
    const Id is_low = IsLowSubgroup(ctx, invocation);
    const Id is_first = IsFirstActiveInvocation(ctx, invocation);
    const Id self_pointer = SelectSharedPointer(ctx, scratch_base, is_low, 0U, sizeof(u32));

    const Id publish_label = ctx.OpLabel();
    const Id publish_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(publish_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_first, publish_label, publish_merge_label);
    ctx.AddLabel(publish_label);
    AtomicStore(ctx, self_pointer, local_word);
    ctx.OpBranch(publish_merge_label);
    ctx.AddLabel(publish_merge_label);

    // The lockstep block dispatcher guarantees that the complete 64-thread guest wave reaches
    // this site in one iteration. Publish both host ballot words before selecting the guest word.
    EmitBarrier(ctx);
    const Id guest_word = AtomicLoad(
        ctx, SharedPointer(ctx, scratch_base, word == 0U ? 0U : sizeof(u32)));
    EmitBarrier(ctx);
    return ctx.OpSelect(ctx.U32[1], is_thread_bit, guest_word, scalar);
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
    const Id active_u32 = ctx.OpSelect(ctx.U32[1], SubgroupAny(ctx, active),
                                       ctx.ConstU32(1U), ctx.u32_zero_value);

    const Id first_label = ctx.OpLabel();
    const Id merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_first, first_label, merge_label);
    ctx.AddLabel(first_label);
    AtomicStore(ctx, active_pointer, active_u32);
    AtomicStore(ctx, generation_pointer, ctx.u32_zero_value);
    AtomicStore(ctx, candidate_generation_pointer, ctx.u32_zero_value);
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
    ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), invocation);
    EmitBarrier(ctx);
}

void EmitWave64LaneRetirementSync(EmitContext& ctx, Id continues, Id invocation,
                                  Id scratch_base) {
    const Id is_low = IsLowSubgroup(ctx, invocation);
    const Id is_first = IsFirstActiveInvocation(ctx, invocation);
    const Id continues_u32 = ctx.OpSelect(ctx.U32[1], SubgroupAny(ctx, continues),
                                          ctx.ConstU32(1U), ctx.u32_zero_value);
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
    const Id next_generation =
        ctx.OpIAdd(ctx.U32[1], AtomicLoad(ctx, self_generation), ctx.ConstU32(1U));
    AtomicStore(ctx, self_active, continues_u32);
    AtomicStore(ctx, self_generation, next_generation);
    ctx.OpBranch(first_merge_label);
    ctx.AddLabel(first_merge_label);
    ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), invocation);

    const Id generation = AtomicLoad(ctx, self_generation);
    const Id peer_was_active = ctx.OpINotEqual(ctx.U1[1], AtomicLoad(ctx, peer_active),
                                               ctx.u32_zero_value);
    const Id wait_label = ctx.OpLabel();
    const Id wait_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(wait_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(peer_was_active, wait_label, wait_merge_label);
    ctx.AddLabel(wait_label);
    SpinUntilAtLeast(ctx, peer_generation, generation);
    ctx.OpBranch(wait_merge_label);
    ctx.AddLabel(wait_merge_label);
}

Id EmitWave64ReadFirstLane(EmitContext& ctx, Id value, Id active, Id invocation,
                           Id scratch_base) {
    const Id native_result = SelectFirstActiveValue(ctx, value, active).value;
    const Id is_low = IsLowSubgroup(ctx, invocation);
    const Id is_first = IsFirstActiveInvocation(ctx, invocation);
    const Id self_generation =
        SelectSharedPointer(ctx, scratch_base, is_low, LowGenerationOffset, HighGenerationOffset);
    const Id peer_active_pointer =
        SelectSharedPointer(ctx, scratch_base, is_low, HighActiveOffset, LowActiveOffset);
    const Id generation = AtomicLoad(ctx, self_generation);
    const Id peer_active = ctx.OpINotEqual(ctx.U1[1], AtomicLoad(ctx, peer_active_pointer),
                                           ctx.u32_zero_value);

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

Id EmitWave64ReadFirstActiveLane(EmitContext& ctx, Id value, Id active, Id invocation,
                                 Id scratch_base) {
    constexpr u32 LowActive = 0U;
    constexpr u32 LowCandidate = sizeof(u32);
    constexpr u32 HighActive = 2U * sizeof(u32);
    constexpr u32 HighCandidate = 3U * sizeof(u32);

    const FirstActiveValue local = SelectFirstActiveValue(ctx, value, active);
    const Id is_low = IsLowSubgroup(ctx, invocation);
    const Id is_first = IsFirstActiveInvocation(ctx, invocation);
    const Id active_pointer =
        SelectSharedPointer(ctx, scratch_base, is_low, LowActive, HighActive);
    const Id candidate_pointer =
        SelectSharedPointer(ctx, scratch_base, is_low, LowCandidate, HighCandidate);

    const Id publish_label = ctx.OpLabel();
    const Id publish_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(publish_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_first, publish_label, publish_merge_label);
    ctx.AddLabel(publish_label);
    AtomicStore(ctx, active_pointer,
                ctx.OpSelect(ctx.U32[1], local.has_active, ctx.ConstU32(1U),
                             ctx.u32_zero_value));
    AtomicStore(ctx, candidate_pointer, local.value);
    ctx.OpBranch(publish_merge_label);
    ctx.AddLabel(publish_merge_label);

    // These operations are emitted only at a workgroup-converged scalar instruction. Each host
    // half publishes the value from its first guest-EXEC-active lane; the low half has priority,
    // exactly matching a 64-lane GCN find-first-set. A closing barrier makes the four words safe
    // to reuse at the next ReadFirstLane site.
    EmitBarrier(ctx);
    const Id low_has_active = ctx.OpINotEqual(
        ctx.U1[1], AtomicLoad(ctx, SharedPointer(ctx, scratch_base, LowActive)),
        ctx.u32_zero_value);
    const Id low_candidate =
        AtomicLoad(ctx, SharedPointer(ctx, scratch_base, LowCandidate));
    const Id high_candidate =
        AtomicLoad(ctx, SharedPointer(ctx, scratch_base, HighCandidate));
    const Id result =
        ctx.OpSelect(ctx.U32[1], low_has_active, low_candidate, high_candidate);
    EmitBarrier(ctx);
    return result;
}

Id EmitWave64MaterializeScalarMaskWord(EmitContext& ctx, Id is_thread_bit, Id bit, Id scalar,
                                      u32 word, Id invocation, Id scratch_base) {
    ASSERT(word < 2U);
    const Id ballot = EmitBallot(ctx, bit);
    const Id local_ballot = ctx.OpCompositeExtract(ctx.U32[1], ballot, 0U);
    const Id local_result =
        ctx.OpSelect(ctx.U32[1], is_thread_bit, local_ballot, scalar);
    const Id is_low = IsLowSubgroup(ctx, invocation);
    const Id owns_word =
        word == 0U ? is_low : ctx.OpLogicalNot(ctx.U1[1], is_low);
    const Id is_first = IsFirstActiveInvocation(ctx, invocation);
    const Id self_generation =
        SelectSharedPointer(ctx, scratch_base, is_low, LowGenerationOffset,
                            HighGenerationOffset);
    const Id peer_active_pointer =
        SelectSharedPointer(ctx, scratch_base, is_low, HighActiveOffset, LowActiveOffset);
    const Id generation = AtomicLoad(ctx, self_generation);
    const Id peer_active = ctx.OpINotEqual(ctx.U1[1], AtomicLoad(ctx, peer_active_pointer),
                                           ctx.u32_zero_value);
    const Id candidate_pointer = SharedPointer(ctx, scratch_base, CandidateOffset);
    const Id candidate_generation_pointer =
        SharedPointer(ctx, scratch_base, CandidateGenerationOffset);

    const Id source_label = ctx.OpLabel();
    const Id receiver_label = ctx.OpLabel();
    const Id final_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(final_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(owns_word, source_label, receiver_label);

    ctx.AddLabel(source_label);
    const Id publish_label = ctx.OpLabel();
    const Id publish_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(publish_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(peer_active, publish_label, publish_merge_label);
    ctx.AddLabel(publish_label);
    const Id first_label = ctx.OpLabel();
    const Id first_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(first_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_first, first_label, first_merge_label);
    ctx.AddLabel(first_label);
    AtomicStore(ctx, candidate_pointer, local_result);
    AtomicStore(ctx, candidate_generation_pointer, generation);
    ctx.OpBranch(first_merge_label);
    ctx.AddLabel(first_merge_label);
    ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), local_result);
    ctx.OpBranch(publish_merge_label);
    ctx.AddLabel(publish_merge_label);
    const Id source_exit_label = ctx.last_label;
    ctx.OpBranch(final_merge_label);

    ctx.AddLabel(receiver_label);
    const Id receive_label = ctx.OpLabel();
    const Id no_source_label = ctx.OpLabel();
    const Id receiver_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(receiver_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(peer_active, receive_label, no_source_label);
    ctx.AddLabel(receive_label);
    SpinUntilAtLeast(ctx, candidate_generation_pointer, generation);
    const Id received = AtomicLoad(ctx, candidate_pointer);
    const Id received_result =
        ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), received);
    const Id receive_exit_label = ctx.last_label;
    ctx.OpBranch(receiver_merge_label);
    ctx.AddLabel(no_source_label);
    const Id inactive_source_result =
        ctx.OpSelect(ctx.U32[1], is_thread_bit, ctx.u32_zero_value, scalar);
    const Id no_source_exit_label = ctx.last_label;
    ctx.OpBranch(receiver_merge_label);
    ctx.AddLabel(receiver_merge_label);
    const Id receiver_result =
        ctx.OpPhi(ctx.U32[1], received_result, receive_exit_label,
                  inactive_source_result, no_source_exit_label);
    const Id receiver_exit_label = ctx.last_label;
    ctx.OpBranch(final_merge_label);

    ctx.AddLabel(final_merge_label);
    return ctx.OpPhi(ctx.U32[1], local_result, source_exit_label, receiver_result,
                     receiver_exit_label);
}

} // namespace Shader::Backend::SPIRV
