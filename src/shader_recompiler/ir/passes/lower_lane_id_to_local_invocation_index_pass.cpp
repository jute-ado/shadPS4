// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

#include "common/alignment.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/profile.h"

namespace Shader::Optimization {

namespace {

constexpr u64 GuestWaveSize = 64;
constexpr u32 GuestWaveScratchBytes = static_cast<u32>(GuestWaveSize * sizeof(u32));

struct LinearInvocationStrides {
    u32 size_x;
    u32 size_xy;
};

std::optional<LinearInvocationStrides> GetGuestWave64Strides(
    const std::array<u32, 3>& workgroup_size) {
    u64 invocation_count = 1;
    for (const u32 dimension : workgroup_size) {
        // Bound each multiplication by the only accepted result. Besides rejecting invalid
        // shapes early, this prevents a three-dimensional u32 product from wrapping through u64
        // and masquerading as a 64-invocation workgroup.
        if (dimension == 0 || invocation_count > GuestWaveSize / dimension) {
            return std::nullopt;
        }
        invocation_count *= dimension;
    }
    if (invocation_count != GuestWaveSize) {
        return std::nullopt;
    }
    // The checked exact-64 product proves both strides fit in u32.
    return LinearInvocationStrides{
        .size_x = workgroup_size[0],
        .size_xy = workgroup_size[0] * workgroup_size[1],
    };
}

bool IsLive(const IR::Inst& inst) {
    return inst.HasUses();
}

bool IsSupportedCrossLaneOperation(const IR::Inst& inst) {
    if (!IsLive(inst)) {
        return false;
    }
    switch (inst.GetOpcode()) {
    case IR::Opcode::ReadFirstLane:
    case IR::Opcode::ReadLane:
    case IR::Opcode::Ballot:
        return true;
    default:
        return false;
    }
}

bool IsLiveUnsupportedCrossLaneOperation(const IR::Inst& inst) {
    if (!inst.HasUses()) {
        return false;
    }
    switch (inst.GetOpcode()) {
    case IR::Opcode::WarpId:
    case IR::Opcode::QuadShuffle:
    case IR::Opcode::WriteLane:
    case IR::Opcode::BallotFindLsb:
    case IR::Opcode::GroupAny:
        return true;
    default:
        return false;
    }
}

enum class Definedness : u8 {
    Visiting,
    Defined,
    Undefined,
};

bool IsDefinitelyDefined(const IR::Value& value,
                         std::unordered_map<const IR::Inst*, Definedness>& memo) {
    if (value.IsEmpty()) {
        return false;
    }
    if (value.IsImmediate()) {
        return true;
    }
    const IR::Inst* const definition = value.TryInstRecursive();
    if (definition == nullptr) {
        // Register, attribute, and other symbolic operands are definitions, not SSA undef values.
        return true;
    }
    if (const auto it = memo.find(definition); it != memo.end()) {
        // A loop-carried phi is defined when every non-cyclic input is defined. SSA cleanup turns a
        // cycle with no defining input into Undef, so revisiting the active chain is provisionally
        // safe while the remaining operands are checked.
        return it->second != Definedness::Undefined;
    }
    switch (definition->GetOpcode()) {
    case IR::Opcode::UndefU1:
    case IR::Opcode::UndefU8:
    case IR::Opcode::UndefU16:
    case IR::Opcode::UndefU32:
    case IR::Opcode::UndefU64:
        memo.emplace(definition, Definedness::Undefined);
        return false;
    default:
        break;
    }

    memo.emplace(definition, Definedness::Visiting);
    for (size_t index = 0; index < definition->NumArgs(); ++index) {
        if (!IsDefinitelyDefined(definition->Arg(index), memo)) {
            memo.insert_or_assign(definition, Definedness::Undefined);
            return false;
        }
    }
    memo.insert_or_assign(definition, Definedness::Defined);
    return true;
}

bool HasUndefinedBallotPredicate(const IR::Inst& inst) {
    if (inst.GetOpcode() != IR::Opcode::Ballot || !inst.HasUses()) {
        return false;
    }
    std::unordered_map<const IR::Inst*, Definedness> memo;
    return !IsDefinitelyDefined(inst.Arg(0), memo);
}

bool IsScalarWordBit(const IR::Value& value) {
    const IR::Inst* const not_equal = value.TryInstRecursive();
    if (not_equal == nullptr || not_equal->GetOpcode() != IR::Opcode::INotEqual32) {
        return false;
    }
    const IR::Inst* const masked = not_equal->Arg(0).TryInstRecursive();
    if (masked == nullptr || masked->GetOpcode() != IR::Opcode::BitwiseAnd32) {
        return false;
    }
    const IR::Inst* const shifted = masked->Arg(0).TryInstRecursive();
    return shifted != nullptr && shifted->GetOpcode() == IR::Opcode::ShiftRightLogical32;
}

bool IsScalarWordReconstructionBallot(const IR::Inst& inst) {
    if (inst.GetOpcode() != IR::Opcode::Ballot || !inst.HasUses()) {
        return false;
    }
    const IR::Value predicate = inst.Arg(0);
    if (IsScalarWordBit(predicate)) {
        return true;
    }
    const IR::Inst* const select = predicate.TryInstRecursive();
    return select != nullptr && select->GetOpcode() == IR::Opcode::SelectU1 &&
           IsScalarWordBit(select->Arg(2));
}

bool HasUnsupportedOperation(const IR::Inst& inst) {
    if (HasUndefinedBallotPredicate(inst)) {
        // Atomically materializing an undefined predicate would turn unspecified subgroup state
        // into a concrete guest-wave mask. Preserve the native path until predicate provenance is
        // established by an earlier IR repair.
        return true;
    }
    if (IsScalarWordReconstructionBallot(inst)) {
        // SSA mask provenance can reconstruct an SGPR dword by balloting its scalar bits. The
        // direct form appears after constant folding; a mixed scalar/thread representation keeps
        // the scalar-bit expression in Select's false arm. Neither form has proven wave-consistent
        // semantics on a subgroup32 host, so keep the native path fail-closed.
        return true;
    }
    switch (inst.GetOpcode()) {
    case IR::Opcode::DataAppend:
    case IR::Opcode::DataConsume:
        // Resource tracking normally lowers these operations. If one survives to this point,
        // preserve the native lane model instead of assuming how its counter is partitioned.
        return true;
    default:
        return IsLiveUnsupportedCrossLaneOperation(inst);
    }
}

bool IsGuestExecStateOperation(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::GetExec:
    case IR::Opcode::SetExec:
        return true;
    default:
        return false;
    }
}

bool HasNonU32SharedMemoryOperation(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadSharedU16:
    case IR::Opcode::WriteSharedU16:
    case IR::Opcode::LoadSharedU64:
    case IR::Opcode::WriteSharedU64:
    case IR::Opcode::SharedAtomicIAdd64:
    case IR::Opcode::SharedAtomicISub64:
    case IR::Opcode::SharedAtomicSMin64:
    case IR::Opcode::SharedAtomicUMin64:
    case IR::Opcode::SharedAtomicSMax64:
    case IR::Opcode::SharedAtomicUMax64:
    case IR::Opcode::SharedAtomicInc64:
    case IR::Opcode::SharedAtomicDec64:
    case IR::Opcode::SharedAtomicAnd64:
    case IR::Opcode::SharedAtomicOr64:
    case IR::Opcode::SharedAtomicXor64:
        return true;
    default:
        return false;
    }
}

bool IsUniformStraightLineProgram(const IR::Program& program) {
    using Type = IR::AbstractSyntaxNode::Type;
    return program.blocks.size() == 1 && program.syntax_list.size() == 2 &&
           program.syntax_list[0].type == Type::Block &&
           program.syntax_list[0].data.block == program.blocks.front() &&
           program.syntax_list[1].type == Type::Return;
}

bool ContainsBlock(std::span<IR::Block* const> blocks, const IR::Block* expected) {
    return std::ranges::find(blocks, expected) != blocks.end();
}

bool HasExactSelectionEdges(const IR::Block& header, const IR::Block& body,
                            const IR::Block& merge) {
    const auto header_successors = header.ImmSuccessors();
    const auto body_predecessors = body.ImmPredecessors();
    const auto body_successors = body.ImmSuccessors();
    const auto merge_predecessors = merge.ImmPredecessors();
    return header_successors.size() == 2 && ContainsBlock(header_successors, &body) &&
           ContainsBlock(header_successors, &merge) && body_predecessors.size() == 1 &&
           body_predecessors.front() == &header && body_successors.size() == 1 &&
           body_successors.front() == &merge && merge_predecessors.size() == 2 &&
           ContainsBlock(merge_predecessors, &header) && ContainsBlock(merge_predecessors, &body);
}

bool HasExistingBarrierBrackets(const IR::Block& block) {
    bool saw_barrier{};
    bool saw_lane_operation{};
    bool barrier_before_first_operation{};
    bool barrier_after_last_operation{};
    for (const IR::Inst& inst : block) {
        if (inst.GetOpcode() == IR::Opcode::Barrier) {
            saw_barrier = true;
            if (saw_lane_operation) {
                barrier_after_last_operation = true;
            }
            continue;
        }
        const bool is_lane_operation = (inst.GetOpcode() == IR::Opcode::LaneId && IsLive(inst)) ||
                                       IsSupportedCrossLaneOperation(inst);
        if (!is_lane_operation) {
            continue;
        }
        if (!saw_lane_operation) {
            barrier_before_first_operation = saw_barrier;
            saw_lane_operation = true;
        }
        // A later lane operation needs a later barrier; an earlier one cannot close the region.
        barrier_after_last_operation = false;
    }
    return saw_lane_operation && barrier_before_first_operation && barrier_after_last_operation;
}

bool IsBarrierBracketedStructuredMerge(const IR::Program& program,
                                       const IR::Block* operation_block) {
    if (operation_block == nullptr || !HasExistingBarrierBrackets(*operation_block)) {
        return false;
    }

    using Type = IR::AbstractSyntaxNode::Type;
    size_t return_count{};
    for (size_t index = 0; index < program.syntax_list.size(); ++index) {
        const Type type = program.syntax_list[index].type;
        switch (type) {
        case Type::Loop:
        case Type::Repeat:
        case Type::Break:
        case Type::Unreachable:
            return false;
        case Type::Return:
            ++return_count;
            if (index + 1 != program.syntax_list.size()) {
                return false;
            }
            break;
        default:
            break;
        }
    }
    if (return_count != 1) {
        return false;
    }

    // Admit only a simple, structured selection whose body and skipped path both reconverge at
    // the candidate block. Other selections may precede or follow it, but the proven region itself
    // must retain the frontend's exact Block/If/Block/EndIf/Block shape.
    for (size_t index = 0; index + 4 < program.syntax_list.size(); ++index) {
        const auto& header_node = program.syntax_list[index];
        const auto& if_node = program.syntax_list[index + 1];
        const auto& body_node = program.syntax_list[index + 2];
        const auto& end_if_node = program.syntax_list[index + 3];
        const auto& merge_node = program.syntax_list[index + 4];
        if (header_node.type != Type::Block || if_node.type != Type::If ||
            body_node.type != Type::Block || end_if_node.type != Type::EndIf ||
            merge_node.type != Type::Block || merge_node.data.block != operation_block ||
            if_node.data.if_node.body != body_node.data.block ||
            if_node.data.if_node.merge != operation_block ||
            end_if_node.data.end_if.merge != operation_block) {
            continue;
        }
        if (HasExactSelectionEdges(*header_node.data.block, *body_node.data.block,
                                   *operation_block)) {
            return true;
        }
    }
    return false;
}

IR::U32 EmitLinearInvocationIndex(IR::IREmitter& ir, const LinearInvocationStrides& strides) {
    const IR::U32 x = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U32 y = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1);
    const IR::U32 z = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 2);
    const IR::U32 y_offset = ir.IMul(y, ir.Imm32(strides.size_x));
    const IR::U32 z_offset = ir.IMul(z, ir.Imm32(strides.size_xy));
    return ir.IAdd(ir.IAdd(x, y_offset), z_offset);
}

IR::U32 EmitScratchAddress(IR::IREmitter& ir, u32 scratch_base, const IR::U32& index) {
    return ir.IAdd(ir.Imm32(scratch_base), ir.IMul(index, ir.Imm32(static_cast<u32>(sizeof(u32)))));
}

void LowerReadLane(IR::Inst& inst, const LinearInvocationStrides& strides, u32 scratch_base) {
    IR::IREmitter ir{*inst.GetParent(), IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::U32 invocation = EmitLinearInvocationIndex(ir, strides);
    ir.WriteShared(32, inst.Arg(0), EmitScratchAddress(ir, scratch_base, invocation));
    ir.Barrier();

    const IR::U32 requested_lane =
        inst.GetOpcode() == IR::Opcode::ReadFirstLane ? ir.Imm32(0) : IR::U32{inst.Arg(1)};
    const IR::U32 guest_lane{
        ir.BitwiseAnd(requested_lane, ir.Imm32(static_cast<u32>(GuestWaveSize - 1)))};
    const IR::U32 value{ir.LoadShared(32, false, EmitScratchAddress(ir, scratch_base, guest_lane))};
    // No invocation may reuse the scratch range until every invocation has consumed this value.
    ir.Barrier();
    inst.ReplaceUsesWithAndRemove(value);
}

void LowerBallot(IR::Inst& inst, const LinearInvocationStrides& strides, u32 scratch_base) {
    IR::IREmitter ir{*inst.GetParent(), IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::U32 low_address = ir.Imm32(scratch_base);
    const IR::U32 high_address = ir.Imm32(scratch_base + static_cast<u32>(sizeof(u32)));

    // Concurrent ordinary zero stores would be a workgroup data race. Atomic AND makes reset
    // deterministic even though every invocation participates.
    static_cast<void>(ir.SharedAtomicAnd(low_address, ir.Imm32(0), false));
    static_cast<void>(ir.SharedAtomicAnd(high_address, ir.Imm32(0), false));
    ir.Barrier();

    const IR::U32 invocation = EmitLinearInvocationIndex(ir, strides);
    const IR::U32 word_lane{ir.BitwiseAnd(invocation, ir.Imm32(31))};
    const IR::U32 lane_bit{ir.ShiftLeftLogical(ir.Imm32(1), word_lane)};
    const IR::U1 predicate{inst.Arg(0)};
    const IR::U1 is_high = ir.IGreaterThanEqual(invocation, ir.Imm32(32), false);
    const IR::U32 low_value{
        ir.Select(ir.LogicalAnd(predicate, ir.LogicalNot(is_high)), lane_bit, ir.Imm32(0))};
    const IR::U32 high_value{ir.Select(ir.LogicalAnd(predicate, is_high), lane_bit, ir.Imm32(0))};
    static_cast<void>(ir.SharedAtomicOr(low_address, low_value, false));
    static_cast<void>(ir.SharedAtomicOr(high_address, high_value, false));
    ir.Barrier();

    const IR::U32 low{ir.LoadShared(32, false, low_address)};
    const IR::U32 high{ir.LoadShared(32, false, high_address)};
    const IR::Value ballot = ir.CompositeConstruct(low, high, ir.Imm32(0), ir.Imm32(0));
    ir.Barrier();
    inst.ReplaceUsesWithAndRemove(ballot);
}

} // Anonymous namespace

void LowerLaneIdToLocalInvocationIndexPass(IR::Program& program, const RuntimeInfo& runtime_info,
                                           const Profile& profile) {
    if (program.info.stage != Stage::Compute || profile.subgroup_size == 0 ||
        profile.subgroup_size >= GuestWaveSize) {
        return;
    }

    const auto strides = GetGuestWave64Strides(runtime_info.cs_info.workgroup_size);
    if (!strides) {
        return;
    }

    bool has_live_lane_id = false;
    bool has_supported_cross_lane_operation = false;
    bool has_guest_exec_state = false;
    bool has_non_u32_shared_memory = false;
    bool lane_operations_span_blocks = false;
    IR::Block* lane_operation_block{};
    std::vector<IR::Inst*> cross_lane_operations;
    for (IR::Block* const block : program.blocks) {
        bool expect_initial_exec = false;
        bool is_first_instruction = block == program.blocks.front();
        for (IR::Inst& inst : *block) {
            if (is_first_instruction && inst.GetOpcode() == IR::Opcode::Prologue) {
                // The frontend initializes EXEC to all lanes immediately after the prologue.
                // This is not a guest EXEC mutation and is the only SetExec accepted here.
                expect_initial_exec = true;
                is_first_instruction = false;
            } else {
                const bool is_initial_exec = expect_initial_exec &&
                                             inst.GetOpcode() == IR::Opcode::SetExec &&
                                             inst.Arg(0).IsImmediate() && inst.Arg(0).U1();
                if (IsGuestExecStateOperation(inst) && !is_initial_exec) {
                    has_guest_exec_state = true;
                }
                expect_initial_exec = false;
                is_first_instruction = false;
            }
            if (HasUnsupportedOperation(inst)) {
                return;
            }
            const bool is_live_lane_id = inst.GetOpcode() == IR::Opcode::LaneId && inst.HasUses();
            const bool is_supported_cross_lane = IsSupportedCrossLaneOperation(inst);
            has_live_lane_id |= is_live_lane_id;
            if (is_live_lane_id || is_supported_cross_lane) {
                if (lane_operation_block == nullptr) {
                    lane_operation_block = block;
                } else if (lane_operation_block != block) {
                    lane_operations_span_blocks = true;
                }
            }
            if (is_supported_cross_lane) {
                has_supported_cross_lane_operation = true;
                cross_lane_operations.push_back(&inst);
            }
            has_non_u32_shared_memory |= HasNonU32SharedMemoryOperation(inst);
        }
    }
    if (!has_live_lane_id && !has_supported_cross_lane_operation) {
        return;
    }

    u32 scratch_base{};
    if (has_supported_cross_lane_operation) {
        const bool has_proven_control_shape =
            IsUniformStraightLineProgram(program) ||
            (!lane_operations_span_blocks &&
             IsBarrierBracketedStructuredMerge(program, lane_operation_block));
        if (!has_proven_control_shape || has_guest_exec_state ||
            (has_non_u32_shared_memory && !profile.supports_workgroup_explicit_memory_layout)) {
            return;
        }
        const u64 guest_shared_memory = runtime_info.cs_info.shared_memory_size;
        const u64 aligned_guest_shared_memory = Common::AlignUp(guest_shared_memory, sizeof(u32));
        const u64 total_shared_memory = aligned_guest_shared_memory + GuestWaveScratchBytes;
        if (total_shared_memory > profile.max_shared_memory_size ||
            total_shared_memory > std::numeric_limits<u32>::max()) {
            return;
        }
        scratch_base = static_cast<u32>(aligned_guest_shared_memory);
        program.info.workgroup_scratch_memory_size =
            static_cast<u32>(total_shared_memory - guest_shared_memory);

        for (IR::Inst* const inst : cross_lane_operations) {
            if (inst->GetOpcode() == IR::Opcode::Ballot) {
                LowerBallot(*inst, *strides, scratch_base);
            } else {
                LowerReadLane(*inst, *strides, scratch_base);
            }
        }
    }

    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : *block) {
            if (inst.GetOpcode() != IR::Opcode::LaneId || !inst.HasUses()) {
                continue;
            }
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            const IR::U32 linear_index = EmitLinearInvocationIndex(ir, *strides);
            inst.ReplaceUsesWithAndRemove(linear_index);
        }
    }
}

} // namespace Shader::Optimization
