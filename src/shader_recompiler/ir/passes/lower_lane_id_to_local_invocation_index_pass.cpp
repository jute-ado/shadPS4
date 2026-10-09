// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/alignment.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/passes/partial_wave_neutral_reads.h"
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

bool HasUndefinedBallotPredicate(const IR::Inst& inst) {
    if (inst.GetOpcode() != IR::Opcode::Ballot || !inst.HasUses()) {
        return false;
    }
    // TryInstRecursive also unwraps Identity nodes, covering the directly equivalent form without
    // inventing broader mask provenance on this clean base.
    const IR::Inst* const predicate = inst.Arg(0).TryInstRecursive();
    return predicate != nullptr && predicate->GetOpcode() == IR::Opcode::UndefU1;
}

bool HasUnsupportedOperation(const IR::Inst& inst) {
    if (HasUndefinedBallotPredicate(inst)) {
        // Atomically materializing an undefined predicate would turn unspecified subgroup state
        // into a concrete guest-wave mask. Preserve the native path until predicate provenance is
        // established by an earlier IR repair.
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

bool IsUniformEntryBlock(const IR::Program& program, const IR::Block* operation_block) {
    using Type = IR::AbstractSyntaxNode::Type;
    return operation_block != nullptr && !program.blocks.empty() &&
           operation_block == program.blocks.front() && !program.syntax_list.empty() &&
           program.syntax_list.front().type == Type::Block &&
           program.syntax_list.front().data.block == operation_block;
}

bool AreBlockLaneValuesConfined(const IR::Program& program, const IR::Block& region) {
    std::unordered_set<const IR::Inst*> syntax_conditions;
    using Type = IR::AbstractSyntaxNode::Type;
    for (const auto& node : program.syntax_list) {
        IR::Value condition;
        switch (node.type) {
        case Type::If:
            condition = node.data.if_node.cond;
            break;
        case Type::Repeat:
            condition = node.data.repeat.cond;
            break;
        case Type::Break:
            condition = node.data.break_node.cond;
            break;
        default:
            continue;
        }
        if (const auto* inst = condition.TryInstRecursive()) {
            syntax_conditions.insert(inst);
        }
    }

    // ASL conditions are external references, not ordinary IR uses. Do not let region-local
    // lane arithmetic change later native-subgroup participation or selectors. A selected
    // cross-lane operation is the boundary: its lowered guest-wave result may escape normally.
    constexpr size_t MaxNodes = 4096;
    std::unordered_set<const IR::Inst*> visited;
    std::vector<const IR::Inst*> pending;
    for (const IR::Inst& inst : region) {
        if (inst.GetOpcode() == IR::Opcode::LaneId && inst.HasUses()) {
            if (visited.size() == MaxNodes) {
                return false;
            }
            visited.insert(&inst);
            pending.push_back(&inst);
        }
    }
    while (!pending.empty()) {
        const IR::Inst* const inst = pending.back();
        pending.pop_back();
        if (inst->GetParent() != &region) {
            return false;
        }
        if (IsSupportedCrossLaneOperation(*inst)) {
            continue;
        }
        if (syntax_conditions.contains(inst) || inst->MayHaveSideEffects() ||
            IsGuestExecStateOperation(*inst)) {
            return false;
        }
        for (const auto& use : inst->Uses()) {
            if (visited.contains(use.user)) {
                continue;
            }
            if (visited.size() == MaxNodes) {
                return false;
            }
            visited.insert(use.user);
            pending.push_back(use.user);
        }
    }
    return true;
}

bool ContainsBlock(std::span<IR::Block* const> blocks, const IR::Block* expected) {
    return std::ranges::find(blocks, expected) != blocks.end();
}

bool HaveMaterializedLaneInputs(std::span<IR::Inst* const> operations,
                                std::span<IR::Block* const> rewritten_blocks) {
    const std::unordered_set<const IR::Inst*> selected(operations.begin(), operations.end());
    std::unordered_set<const IR::Inst*> visited;
    std::vector<const IR::Inst*> pending;
    constexpr size_t MaxNodes = 4096;
    const auto enqueue = [&](const IR::Value& value) {
        const auto* const inst = value.TryInstRecursive();
        if (inst == nullptr || selected.contains(inst) || visited.contains(inst)) {
            return true;
        }
        if (visited.size() == MaxNodes) {
            return false;
        }
        visited.insert(inst);
        pending.push_back(inst);
        return true;
    };
    // A selected cross-lane result is a guest-wave boundary, but an external native LaneId
    // is not. Check every root's operands independently before stopping at selected results.
    // This includes Phi arms, transferred values, lane selectors and ballot predicates.
    for (const IR::Inst* const operation : operations) {
        for (size_t arg = 0; arg < operation->NumArgs(); ++arg) {
            if (!enqueue(operation->Arg(arg))) {
                return false;
            }
        }
    }
    while (!pending.empty()) {
        const IR::Inst* const inst = pending.back();
        pending.pop_back();
        if (IsSupportedCrossLaneOperation(*inst)) {
            // Selected results were stopped at enqueue. An unselected native subgroup result
            // is not a guest-wave value even when it has no explicit LaneId in its ancestry.
            return false;
        }
        if (inst->GetOpcode() == IR::Opcode::LaneId) {
            if (!ContainsBlock(rewritten_blocks, inst->GetParent())) {
                return false;
            }
            continue;
        }
        for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
            if (!enqueue(inst->Arg(arg))) {
                return false;
            }
        }
    }
    return true;
}

bool IsProvenUniformValue(const IR::Value& value, const IR::Block& source_block,
                         std::span<IR::Inst* const> selected_operations,
                         std::unordered_map<const IR::Inst*, bool>& memo,
                         std::unordered_set<const IR::Inst*>& visiting, u32 depth = 0) {
    if (value.IsImmediate()) {
        return true;
    }
    const IR::Inst* const inst = value.TryInstRecursive();
    if (inst == nullptr || inst->GetParent() != &source_block || depth >= 64U ||
        memo.size() + visiting.size() >= 4096U) {
        return false;
    }
    if (const auto it = memo.find(inst); it != memo.end()) {
        return it->second;
    }
    if (!visiting.insert(inst).second) {
        return false;
    }

    bool uniform = false;
    const bool selected = std::ranges::find(selected_operations, inst) != selected_operations.end();
    switch (inst->GetOpcode()) {
    case IR::Opcode::ReadLane:
    case IR::Opcode::ReadFirstLane: {
        // Being selected for workgroup lowering does not make a varying lane selector uniform.
        // An undefined source is also not a positive proof: a scratch copy is not a freeze.
        const IR::Inst* const source = inst->Arg(0).TryInstRecursive();
        const bool undefined_source = source != nullptr && source->GetOpcode() == IR::Opcode::UndefU32;
        uniform = selected && !undefined_source &&
                  (inst->GetOpcode() == IR::Opcode::ReadFirstLane ||
                   (inst->Arg(1).IsImmediate() && inst->Arg(1).U32() < GuestWaveSize));
        break;
    }
    case IR::Opcode::Ballot:
        uniform = selected;
        break;
    case IR::Opcode::BitwiseOr32:
    case IR::Opcode::IEqual32:
    case IR::Opcode::INotEqual32:
    case IR::Opcode::LogicalAnd:
    case IR::Opcode::LogicalOr:
    case IR::Opcode::LogicalNot:
    case IR::Opcode::ConditionRef:
        uniform = true;
        for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
            uniform &= IsProvenUniformValue(inst->Arg(arg), source_block, selected_operations, memo, visiting,
                                           depth + 1U);
        }
        break;
    default:
        // Memory, native wave operations, Phi and opaque operations are not positive proofs.
        break;
    }
    visiting.erase(inst);
    memo.emplace(inst, uniform);
    return uniform;
}

struct UniformBodySelection {
    IR::Block* block;
    std::vector<IR::Inst*> operations;
    std::vector<IR::Block*> additional_blocks{};
};

std::optional<UniformBodySelection> SelectFirstUniformBody(
    const IR::Program& program, const IR::Block& entry,
    std::span<IR::Inst* const> entry_operations) {
    using Type = IR::AbstractSyntaxNode::Type;
    if (program.syntax_list.size() < 3 || program.syntax_list[1].type != Type::If ||
        program.syntax_list[2].type != Type::Block || !entry.ImmPredecessors().empty()) {
        return std::nullopt;
    }
    const auto& selection = program.syntax_list[1].data.if_node;
    IR::Block* const body = program.syntax_list[2].data.block;
    if (body == nullptr || body == &entry || body != selection.body || selection.merge == nullptr ||
        selection.merge == body || selection.merge == &entry ||
        entry.ImmSuccessors().size() != 2 ||
        !ContainsBlock(entry.ImmSuccessors(), body) ||
        !ContainsBlock(entry.ImmSuccessors(), selection.merge) ||
        body->ImmPredecessors().size() != 1 || body->ImmPredecessors().front() != &entry) {
        return std::nullopt;
    }
    std::unordered_map<const IR::Inst*, bool> memo;
    std::unordered_set<const IR::Inst*> visiting;
    if (!IsProvenUniformValue(selection.cond, entry, entry_operations, memo, visiting) ||
        !AreBlockLaneValuesConfined(program, *body)) {
        return std::nullopt;
    }

    bool full_exec = true;
    const auto update_exec = [&](const IR::Inst& inst) {
        if (inst.GetOpcode() == IR::Opcode::SetExec) {
            full_exec = inst.Arg(0).IsImmediate() && inst.Arg(0).U1();
        }
    };
    for (const IR::Inst& inst : entry) {
        update_exec(inst);
    }
    UniformBodySelection result{body, {}};
    for (IR::Inst& inst : *body) {
        update_exec(inst);
        const bool lane_id = inst.GetOpcode() == IR::Opcode::LaneId && inst.HasUses();
        const bool cross_lane = IsSupportedCrossLaneOperation(inst);
        if ((lane_id || cross_lane) && !full_exec) {
            return std::nullopt;
        }
        if (cross_lane) {
            result.operations.push_back(&inst);
        }
    }
    if (result.operations.empty()) {
        return std::nullopt;
    }
    return result;
}

bool MatchesExpectedEdges(std::span<IR::Block* const> actual,
                          const std::vector<IR::Block*>& expected) {
    return actual.size() == expected.size() &&
           std::ranges::all_of(actual, [&](const IR::Block* block) {
               return ContainsBlock(expected, block);
           });
}

struct ReconvergedPrefix {
    IR::Block* merge;
    size_t syntax_index;
};

std::optional<ReconvergedPrefix> ProveReconvergedPrefix(
    const IR::Program& program, IR::Block* uniform_header, size_t header_index,
    IR::Block* forbidden_outer_merge = nullptr) {
    using Type = IR::AbstractSyntaxNode::Type;
    if (header_index >= program.syntax_list.size() ||
        program.syntax_list.size() - header_index < 3 ||
        program.syntax_list[header_index].type != Type::Block ||
        program.syntax_list[header_index].data.block != uniform_header ||
        program.syntax_list[header_index + 1].type != Type::If) {
        return std::nullopt;
    }
    IR::Block* const candidate = program.syntax_list[header_index + 1].data.if_node.merge;
    if (candidate == nullptr || candidate == uniform_header || candidate == forbidden_outer_merge) {
        return std::nullopt;
    }

    // Reconstruct the closed acyclic selection prefix. A balanced ASL alone is not enough:
    // extra CFG predecessors, bypass edges or backedges would invalidate reconvergence.
    using EdgeMap = std::unordered_map<IR::Block*, std::vector<IR::Block*>>;
    EdgeMap successors;
    EdgeMap predecessors;
    std::unordered_set<IR::Block*> visited{uniform_header};
    std::vector<IR::Block*> merges;
    const auto add_edge = [&](IR::Block* from, IR::Block* to) {
        if (!ContainsBlock(successors[from], to)) {
            successors[from].push_back(to);
            predecessors[to].push_back(from);
        }
    };
    IR::Block* current = uniform_header;
    IR::Block* expected_next{};
    bool closed = false;
    size_t candidate_index{};
    constexpr size_t MaxPrefixNodes = 1024;
    for (size_t index = header_index + 1;
         index < program.syntax_list.size() && index - header_index - 1 < MaxPrefixNodes; ++index) {
        const auto& node = program.syntax_list[index];
        switch (node.type) {
        case Type::If: {
            const auto& selection = node.data.if_node;
            if (current == nullptr || expected_next != nullptr || selection.body == nullptr ||
                selection.merge == nullptr || selection.body == selection.merge ||
                selection.body == forbidden_outer_merge ||
                selection.merge == forbidden_outer_merge ||
                merges.size() >= 64U) {
                return std::nullopt;
            }
            add_edge(current, selection.body);
            add_edge(current, selection.merge);
            merges.push_back(selection.merge);
            current = nullptr;
            expected_next = selection.body;
            break;
        }
        case Type::Block: {
            IR::Block* const block = node.data.block;
            if (block == nullptr || block == forbidden_outer_merge ||
                (expected_next != nullptr && block != expected_next) ||
                !visited.insert(block).second) {
                return std::nullopt;
            }
            if (current != nullptr) {
                add_edge(current, block);
            }
            current = block;
            expected_next = nullptr;
            break;
        }
        case Type::EndIf:
            if (current == nullptr || expected_next != nullptr || merges.empty() ||
                node.data.end_if.merge != merges.back()) {
                return std::nullopt;
            }
            add_edge(current, merges.back());
            expected_next = merges.back();
            merges.pop_back();
            current = nullptr;
            if (merges.empty()) {
                if (expected_next != candidate || index + 1 >= program.syntax_list.size() ||
                    program.syntax_list[index + 1].type != Type::Block ||
                    program.syntax_list[index + 1].data.block != candidate ||
                    !visited.insert(candidate).second) {
                    return std::nullopt;
                }
                closed = true;
                candidate_index = index + 1;
            }
            break;
        default:
            // Loops and possible exits inside this prefix are not part of the proof. Later
            // syntax is deliberately not inspected once the declared merge has been reached.
            return std::nullopt;
        }
        if (closed) {
            break;
        }
    }
    if (!closed) {
        return std::nullopt;
    }
    for (IR::Block* block : visited) {
        if (block != uniform_header &&
            !MatchesExpectedEdges(block->ImmPredecessors(), predecessors[block])) {
            return std::nullopt;
        }
        if (block != candidate &&
            !MatchesExpectedEdges(block->ImmSuccessors(), successors[block])) {
            return std::nullopt;
        }
    }
    return ReconvergedPrefix{candidate, candidate_index};
}

bool HasDefinedReadLaneSource(const IR::Value& source) {
    std::unordered_set<const IR::Inst*> visited;
    std::vector<const IR::Inst*> pending;
    const auto enqueue = [&](const IR::Value& value) {
        const auto* inst = value.TryInstRecursive();
        if (inst == nullptr || visited.contains(inst)) {
            return true;
        }
        if (visited.size() >= 4096U) {
            return false;
        }
        visited.insert(inst);
        pending.push_back(inst);
        return true;
    };
    if (!enqueue(source)) {
        return false;
    }
    while (!pending.empty()) {
        const auto* inst = pending.back();
        pending.pop_back();
        switch (inst->GetOpcode()) {
        case IR::Opcode::UndefU1:
        case IR::Opcode::UndefU8:
        case IR::Opcode::UndefU16:
        case IR::Opcode::UndefU32:
        case IR::Opcode::UndefU64:
            return false;
        default:
            break;
        }
        for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
            if (!enqueue(inst->Arg(arg))) {
                return false;
            }
        }
    }
    return true;
}

std::optional<UniformBodySelection> SelectReconvergedOperations(
    const IR::Program& program, IR::Block* candidate, bool allow_fixed_read_without_full_exec) {
    if (!AreBlockLaneValuesConfined(program, *candidate)) {
        return std::nullopt;
    }

    // Host invocations reconverge at the closed selection merge, but guest EXEC may still be
    // partial. READLANE ignores guest EXEC; only a fixed, defined source may use that exception.
    // Ballot, ReadFirstLane and native lane-id rewrites retain the explicit full-mask requirement.
    bool full_exec = false;
    UniformBodySelection result{candidate, {}};
    for (IR::Inst& inst : *candidate) {
        if (inst.GetOpcode() == IR::Opcode::SetExec) {
            full_exec = inst.Arg(0).IsImmediate() && inst.Arg(0).U1();
        }
        const bool lane_id = inst.GetOpcode() == IR::Opcode::LaneId && inst.HasUses();
        const bool cross_lane = IsSupportedCrossLaneOperation(inst);
        const bool fixed_read_exception =
            allow_fixed_read_without_full_exec && inst.GetOpcode() == IR::Opcode::ReadLane &&
            inst.Arg(1).IsImmediate() && inst.Arg(1).U32() < GuestWaveSize &&
            HasDefinedReadLaneSource(inst.Arg(0));
        if ((lane_id || cross_lane) && !full_exec && !fixed_read_exception) {
            return std::nullopt;
        }
        if (cross_lane) {
            result.operations.push_back(&inst);
        }
    }
    return result;
}

std::optional<UniformBodySelection> SelectFirstReconvergedMerge(
    const IR::Program& program, IR::Block* uniform_header, size_t header_index,
    IR::Block* forbidden_outer_merge = nullptr) {
    const auto proof =
        ProveReconvergedPrefix(program, uniform_header, header_index, forbidden_outer_merge);
    if (!proof) {
        return std::nullopt;
    }
    auto result = SelectReconvergedOperations(program, proof->merge, false);
    return result && !result->operations.empty() ? result : std::nullopt;
}

std::optional<UniformBodySelection> SelectChainedTopLevelMerge(const IR::Program& program,
                                                             IR::Block* entry) {
    IR::Block* header = entry;
    size_t header_index{};
    std::optional<UniformBodySelection> selected;
    // A completed selection can be the full-participation header of another selection. Do not
    // require a lane operation in every intermediate merge, or inspect unrelated later loops.
    for (size_t selections = 0; selections < 64U && header_index < 1024U; ++selections) {
        const auto proof = ProveReconvergedPrefix(program, header, header_index);
        if (!proof || proof->syntax_index >= 1024U) {
            return selected;
        }
        auto result = SelectReconvergedOperations(program, proof->merge, true);
        if (!result) {
            return selected;
        }
        if (!result->operations.empty()) {
            if (!selected) {
                selected = std::move(result);
            } else {
                selected->additional_blocks.push_back(result->block);
                selected->operations.insert(selected->operations.end(), result->operations.begin(),
                                            result->operations.end());
            }
        }
        header = proof->merge;
        header_index = proof->syntax_index;
    }
    return selected;
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

    PartialWave::RecoverNeutralReads(program, runtime_info);

    const auto strides = GetGuestWave64Strides(runtime_info.cs_info.workgroup_size);
    if (!strides) {
        return;
    }

    bool has_live_lane_id = false;
    bool has_supported_cross_lane_operation = false;
    bool has_guest_exec_state = false;
    bool entry_lane_operations_have_full_exec = true;
    bool has_non_u32_shared_memory = false;
    bool lane_operations_span_blocks = false;
    IR::Block* lane_operation_block{};
    std::vector<IR::Inst*> cross_lane_operations;
    std::vector<IR::Inst*> entry_cross_lane_operations;
    IR::Block* const entry_block =
        !program.syntax_list.empty() &&
                program.syntax_list.front().type == IR::AbstractSyntaxNode::Type::Block
            ? program.syntax_list.front().data.block
            : nullptr;
    for (IR::Block* const block : program.blocks) {
        const bool is_uniform_entry = block == entry_block;
        bool entry_exec_is_proven_full = true;
        for (IR::Inst& inst : *block) {
            const bool is_proven_full_exec_set = inst.GetOpcode() == IR::Opcode::SetExec &&
                                                 inst.Arg(0).IsImmediate() && inst.Arg(0).U1();
            if (is_uniform_entry && inst.GetOpcode() == IR::Opcode::SetExec) {
                entry_exec_is_proven_full = is_proven_full_exec_set;
            }
            if (IsGuestExecStateOperation(inst) && !is_proven_full_exec_set) {
                has_guest_exec_state = true;
            }
            if (HasUnsupportedOperation(inst)) {
                return;
            }
            const bool is_live_lane_id = inst.GetOpcode() == IR::Opcode::LaneId && inst.HasUses();
            const bool is_supported_cross_lane = IsSupportedCrossLaneOperation(inst);
            has_live_lane_id |= is_live_lane_id;
            if (is_uniform_entry && (is_live_lane_id || is_supported_cross_lane) &&
                !entry_exec_is_proven_full) {
                entry_lane_operations_have_full_exec = false;
            }
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
                if (is_uniform_entry) {
                    entry_cross_lane_operations.push_back(&inst);
                }
            }
            has_non_u32_shared_memory |= HasNonU32SharedMemoryOperation(inst);
        }
    }
    if (!has_live_lane_id && !has_supported_cross_lane_operation) {
        return;
    }

    u32 scratch_base{};
    bool lower_uniform_entry_only = false;
    bool lower_top_level_merge_only = false;
    IR::Block* selected_uniform_body{};
    IR::Block* selected_uniform_merge{};
    std::vector<IR::Block*> selected_top_level_blocks;
    if (has_supported_cross_lane_operation) {
        lower_uniform_entry_only =
            (!IsUniformStraightLineProgram(program) || has_guest_exec_state) &&
            !entry_cross_lane_operations.empty() &&
            entry_lane_operations_have_full_exec && IsUniformEntryBlock(program, entry_block) &&
            AreBlockLaneValuesConfined(program, *entry_block);
        std::optional<UniformBodySelection> top_level_merge;
        if (!lower_uniform_entry_only && IsUniformEntryBlock(program, entry_block) &&
            entry_block->ImmPredecessors().empty()) {
            // Every invocation reaches the initial selection header, even when its condition
            // is divergent and it contains no lane reads. Only the proven merge is rewritten;
            // native lane values in the header and divergent interior must remain untouched.
            top_level_merge = SelectChainedTopLevelMerge(program, entry_block);
        }
        const bool has_proven_control_shape =
            lower_uniform_entry_only || top_level_merge.has_value() ||
            IsUniformStraightLineProgram(program) ||
            (!lane_operations_span_blocks &&
             IsBarrierBracketedStructuredMerge(program, lane_operation_block));
        const bool selected_region_has_guest_exec =
            lower_uniform_entry_only ? !entry_lane_operations_have_full_exec
                                     : !top_level_merge.has_value() && has_guest_exec_state;
        if (!has_proven_control_shape || selected_region_has_guest_exec ||
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

        if (lower_uniform_entry_only) {
            cross_lane_operations = std::move(entry_cross_lane_operations);
            if (auto body = SelectFirstUniformBody(program, *entry_block, cross_lane_operations)) {
                selected_uniform_body = body->block;
                cross_lane_operations.insert(cross_lane_operations.end(), body->operations.begin(),
                                             body->operations.end());
                if (auto merge = SelectFirstReconvergedMerge(
                        program, body->block, 2, program.syntax_list[1].data.if_node.merge)) {
                    selected_uniform_merge = merge->block;
                    cross_lane_operations.insert(cross_lane_operations.end(),
                                                 merge->operations.begin(), merge->operations.end());
                }
            }
        } else if (top_level_merge) {
            lower_top_level_merge_only = true;
            selected_uniform_merge = top_level_merge->block;
            selected_top_level_blocks.push_back(top_level_merge->block);
            selected_top_level_blocks.insert(selected_top_level_blocks.end(),
                                            top_level_merge->additional_blocks.begin(),
                                            top_level_merge->additional_blocks.end());
            cross_lane_operations = std::move(top_level_merge->operations);
        }
        if (lower_uniform_entry_only || lower_top_level_merge_only) {
            std::vector<IR::Block*> rewritten_blocks;
            if (lower_uniform_entry_only) {
                rewritten_blocks.push_back(entry_block);
            }
            if (selected_uniform_body != nullptr) {
                rewritten_blocks.push_back(selected_uniform_body);
            }
            if (selected_uniform_merge != nullptr) {
                rewritten_blocks.push_back(selected_uniform_merge);
            }
            if (lower_top_level_merge_only) {
                rewritten_blocks = selected_top_level_blocks;
            }
            if (!HaveMaterializedLaneInputs(cross_lane_operations, rewritten_blocks)) {
                return;
            }
        }
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
        if (lower_top_level_merge_only && !ContainsBlock(selected_top_level_blocks, block)) {
            continue;
        }
        if (lower_uniform_entry_only && block != entry_block && block != selected_uniform_body &&
            block != selected_uniform_merge) {
            continue;
        }
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
