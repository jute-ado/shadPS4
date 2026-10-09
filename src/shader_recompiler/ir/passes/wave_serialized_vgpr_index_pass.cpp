// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <optional>
#include <vector>

#include "shader_recompiler/ir/passes/ir_passes.h"

namespace Shader::Optimization {
namespace {

struct IndexedCompare {
    u32 index;
    IR::Inst* compare;
};

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

bool GatesSerializedExecution(const IR::Inst& compare) {
    for (const IR::Use& compare_use : compare.Uses()) {
        if (compare_use.user->GetOpcode() != IR::Opcode::LogicalAnd) {
            continue;
        }
        if (std::ranges::any_of(compare_use.user->Uses(), [](const IR::Use& logical_and_use) {
                return logical_and_use.user->GetOpcode() == IR::Opcode::ConditionRef;
            })) {
            return true;
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
    std::vector<IndexedCompare> indexed_compares;
    for (const IR::Use& use : read_first.Uses()) {
        IR::Inst* const compare = use.user;
        const std::optional<IR::Value> other = OtherComparisonArgument(*compare, read_first);
        if (!other) {
            return false;
        }
        if (*other == per_lane_index) {
            if (selected_lane_compare != nullptr || !GatesSerializedExecution(*compare)) {
                return false;
            }
            selected_lane_compare = compare;
            continue;
        }
        if (!other->IsImmediate() || !IsVgprSelectorCompare(*compare)) {
            return false;
        }
        indexed_compares.push_back(IndexedCompare{
            .index = other->U32(),
            .compare = compare,
        });
    }

    if (selected_lane_compare == nullptr || indexed_compares.size() < 2) {
        return false;
    }
    std::ranges::sort(indexed_compares, {}, &IndexedCompare::index);
    for (u32 index = 0; index < indexed_compares.size(); ++index) {
        if (indexed_compares[index].index != index) {
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

void WaveSerializedVgprIndexPass(IR::Program& program) {
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::ReadFirstLane && inst.HasUses()) {
                TryEliminate(inst);
            }
        }
    }
}

} // namespace Shader::Optimization
