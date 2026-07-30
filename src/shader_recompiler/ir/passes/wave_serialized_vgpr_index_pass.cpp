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

bool FeedsConditionRef(const IR::Inst& inst) {
    return std::ranges::any_of(inst.Uses(), [](const IR::Use& use) {
        return use.user->GetOpcode() == IR::Opcode::ConditionRef;
    });
}

bool GatesSerializedLoop(const IR::Inst& compare) {
    if (std::ranges::distance(compare.Uses()) != 1) {
        return false;
    }
    IR::Inst* const active_selected = compare.Uses().begin()->user;
    if (active_selected->GetOpcode() != IR::Opcode::LogicalAnd ||
        !FeedsConditionRef(*active_selected)) {
        return false;
    }

    for (const IR::Use& active_use : active_selected->Uses()) {
        IR::Inst* const logical_not = active_use.user;
        if (logical_not->GetOpcode() != IR::Opcode::LogicalNot) {
            continue;
        }
        for (const IR::Use& not_use : logical_not->Uses()) {
            IR::Inst* const remaining_lanes = not_use.user;
            if (remaining_lanes->GetOpcode() == IR::Opcode::LogicalAnd &&
                FeedsConditionRef(*remaining_lanes)) {
                return true;
            }
        }
    }
    return false;
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
