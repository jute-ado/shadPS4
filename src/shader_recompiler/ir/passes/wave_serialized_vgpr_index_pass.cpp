// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <optional>
#include <vector>

#include "shader_recompiler/ir/passes/ir_passes.h"

namespace Shader::Optimization {
namespace {

bool IsProducedBy(const IR::Value& value, const IR::Inst& producer) {
    return !value.IsImmediate() && value.TryInstRecursive() == &producer;
}

bool AreEquivalentLoopMasks(IR::Value lhs, IR::Value rhs) {
    lhs = lhs.Resolve();
    rhs = rhs.Resolve();
    if (lhs == rhs) {
        return true;
    }
    if (lhs.IsImmediate() || rhs.IsImmediate()) {
        return false;
    }

    IR::Inst* const lhs_phi = lhs.TryInstRecursive();
    IR::Inst* const rhs_phi = rhs.TryInstRecursive();
    if (lhs_phi == nullptr || rhs_phi == nullptr ||
        lhs_phi->GetOpcode() != IR::Opcode::Phi ||
        rhs_phi->GetOpcode() != IR::Opcode::Phi ||
        lhs_phi->GetParent() != rhs_phi->GetParent() ||
        lhs_phi->Type() != rhs_phi->Type() ||
        lhs_phi->NumArgs() != rhs_phi->NumArgs()) {
        return false;
    }

    for (size_t index = 0; index < lhs_phi->NumArgs(); ++index) {
        if (lhs_phi->PhiBlock(index) != rhs_phi->PhiBlock(index) ||
            lhs_phi->Arg(index).Resolve() != rhs_phi->Arg(index).Resolve()) {
            return false;
        }
    }
    return true;
}

std::optional<IR::Value> OtherComparisonArgument(IR::Inst& compare, const IR::Inst& read_first) {
    if (compare.GetOpcode() != IR::Opcode::IEqual32) {
        return std::nullopt;
    }
    if (IsProducedBy(compare.Arg(0), read_first)) {
        return compare.Arg(1).Resolve();
    }
    if (IsProducedBy(compare.Arg(1), read_first)) {
        return compare.Arg(0).Resolve();
    }
    return std::nullopt;
}

bool IsVgprSelectorCompare(const IR::Inst& compare) {
    if (!compare.HasUses()) {
        return false;
    }
    return std::ranges::all_of(compare.Uses(), [](const IR::Use& use) {
        return use.operand == 0 && use.user->GetOpcode() == IR::Opcode::SelectU32;
    });
}

std::optional<IR::Value> OtherLogicalAndArgument(IR::Inst& logical_and,
                                                 const IR::Inst& producer) {
    if (logical_and.GetOpcode() != IR::Opcode::LogicalAnd) {
        return std::nullopt;
    }
    const bool first_is_producer = IsProducedBy(logical_and.Arg(0), producer);
    const bool second_is_producer = IsProducedBy(logical_and.Arg(1), producer);
    if (first_is_producer == second_is_producer) {
        return std::nullopt;
    }
    return (first_is_producer ? logical_and.Arg(1) : logical_and.Arg(0)).Resolve();
}

bool GatesSerializedLoop(const IR::Inst& compare) {
    if (std::ranges::distance(compare.Uses()) != 1) {
        return false;
    }
    IR::Inst* const active_selected = compare.Uses().begin()->user;
    const std::optional<IR::Value> active_mask =
        OtherLogicalAndArgument(*active_selected, compare);
    if (!active_mask || std::ranges::distance(active_selected->Uses()) != 2) {
        return false;
    }

    bool feeds_selected_condition{};
    IR::Inst* logical_not{};
    for (const IR::Use& active_use : active_selected->Uses()) {
        if (active_use.operand == 0 &&
            active_use.user->GetOpcode() == IR::Opcode::ConditionRef) {
            if (feeds_selected_condition) {
                return false;
            }
            feeds_selected_condition = true;
            continue;
        }
        if (active_use.operand != 0 ||
            active_use.user->GetOpcode() != IR::Opcode::LogicalNot ||
            logical_not != nullptr) {
            return false;
        }
        logical_not = active_use.user;
    }
    if (!feeds_selected_condition || logical_not == nullptr ||
        std::ranges::distance(logical_not->Uses()) != 1) {
        return false;
    }

    IR::Inst* const remaining_lanes = logical_not->Uses().begin()->user;
    const std::optional<IR::Value> remaining_mask =
        OtherLogicalAndArgument(*remaining_lanes, *logical_not);
    if (!remaining_mask || !AreEquivalentLoopMasks(*remaining_mask, *active_mask) ||
        std::ranges::distance(remaining_lanes->Uses()) != 1) {
        return false;
    }

    const IR::Use& remaining_use = *remaining_lanes->Uses().begin();
    return remaining_use.operand == 0 &&
           remaining_use.user->GetOpcode() == IR::Opcode::ConditionRef;
}

bool TryEliminate(IR::Inst& read_first) {
    const IR::Value per_lane_index = read_first.Arg(0).Resolve();
    if (per_lane_index.IsImmediate()) {
        return false;
    }

    IR::Inst* selected_lane_compare{};
    std::vector<u32> selector_indices;
    for (const IR::Use& use : read_first.Uses()) {
        IR::Inst* const compare = use.user;
        const std::optional<IR::Value> other = OtherComparisonArgument(*compare, read_first);
        if (!other) {
            return false;
        }
        if (*other == per_lane_index) {
            if (selected_lane_compare != nullptr || !GatesSerializedLoop(*compare)) {
                return false;
            }
            selected_lane_compare = compare;
            continue;
        }
        if (!other->IsImmediate() || !IsVgprSelectorCompare(*compare)) {
            return false;
        }
        selector_indices.push_back(other->U32());
    }

    if (selected_lane_compare == nullptr || selector_indices.size() < 2) {
        return false;
    }
    std::ranges::sort(selector_indices);
    for (u32 index = 0; index < selector_indices.size(); ++index) {
        if (selector_indices[index] != index) {
            return false;
        }
    }

    // GCN compilers serialize a relative VGPR access by broadcasting one active lane's index,
    // masking EXEC to lanes with that index, then repeating until every lane has run. The SPMD
    // backend has already expanded the relative access into per-invocation selects, so using the
    // original per-lane index executes the same access once for every lane. Making the lane match
    // unconditional also empties the compiler's remaining-lanes mask after that first iteration.
    selected_lane_compare->ReplaceUsesWithAndRemove(IR::Value{true});
    read_first.ReplaceUsesWithAndRemove(per_lane_index);
    return true;
}

} // Anonymous namespace

bool WaveSerializedVgprIndexPass(IR::Program& program) {
    bool changed{};
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::ReadFirstLane && inst.HasUses()) {
                changed |= TryEliminate(inst);
            }
        }
    }
    return changed;
}

} // namespace Shader::Optimization
