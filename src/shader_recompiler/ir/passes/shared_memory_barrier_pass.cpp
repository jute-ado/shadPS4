// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>
#include <optional>
#include <unordered_set>
#include <vector>
#include <queue>
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Optimization {

static bool IsLoadShared(const IR::Inst& inst) {
    return inst.GetOpcode() == IR::Opcode::LoadSharedU16 ||
           inst.GetOpcode() == IR::Opcode::LoadSharedU32 ||
           inst.GetOpcode() == IR::Opcode::LoadSharedU64;
}

static bool IsWriteShared(const IR::Inst& inst) {
    return inst.GetOpcode() == IR::Opcode::WriteSharedU16 ||
           inst.GetOpcode() == IR::Opcode::WriteSharedU32 ||
           inst.GetOpcode() == IR::Opcode::WriteSharedU64;
}

// Inserts barriers when a shared memory write and read occur in the same basic block.
static void EmitBarrierInBlock(IR::Block* block) {
    enum class BarrierAction : u32 {
        None,
        BarrierOnWrite,
        BarrierOnRead,
    };
    BarrierAction action{};
    for (IR::Inst& inst : block->Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::Barrier) {
            action = BarrierAction::None;
            continue;
        }
        if (IsLoadShared(inst)) {
            if (action == BarrierAction::BarrierOnRead) {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                ir.Barrier();
            }
            action = BarrierAction::BarrierOnWrite;
            continue;
        }
        if (IsWriteShared(inst)) {
            if (action == BarrierAction::BarrierOnWrite) {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                ir.Barrier();
            }
            action = BarrierAction::BarrierOnRead;
        }
    }
    if (action != BarrierAction::None) {
        IR::IREmitter ir{*block, --block->end()};
        ir.Barrier();
    }
}

using NodeSet = std::unordered_set<const IR::Block*>;
using InstSet = std::unordered_set<const IR::Inst*>;

struct U32Range {
    u32 min;
    u32 max;
};

// Conservatively bound local invocation arithmetic so expressions such as a flattened invocation
// index masked to a guest-wave base can be proven workgroup-uniform.
static u32 PossibleBits(u32 max) {
    u32 mask{};
    while (max != 0) {
        mask = (mask << 1) | 1U;
        max >>= 1;
    }
    return mask;
}

static std::optional<U32Range> GetWorkgroupU32Range(const IR::Value& value,
                                                    const std::array<u32, 3>& workgroup_size,
                                                    InstSet& visiting) {
    if (value.Type() != IR::Type::U32) {
        return std::nullopt;
    }
    if (value.IsImmediate()) {
        const u32 immediate = value.U32();
        return U32Range{immediate, immediate};
    }

    IR::Inst* const inst = value.InstRecursive();
    if (!visiting.emplace(inst).second) {
        return std::nullopt;
    }

    std::optional<U32Range> result;
    const auto range = [&](size_t arg) {
        return GetWorkgroupU32Range(inst->Arg(arg), workgroup_size, visiting);
    };
    switch (inst->GetOpcode()) {
    case IR::Opcode::GetAttributeU32: {
        if (inst->Arg(0).Attribute() != IR::Attribute::LocalInvocationId ||
            !inst->Arg(1).IsImmediate()) {
            break;
        }
        const u32 component = inst->Arg(1).U32();
        if (component < workgroup_size.size() && workgroup_size[component] != 0) {
            result = U32Range{0, workgroup_size[component] - 1};
        }
        break;
    }
    case IR::Opcode::IAdd32: {
        const auto lhs = range(0);
        const auto rhs = range(1);
        if (!lhs || !rhs) {
            break;
        }
        const u64 min = static_cast<u64>(lhs->min) + rhs->min;
        const u64 max = static_cast<u64>(lhs->max) + rhs->max;
        if (max <= std::numeric_limits<u32>::max()) {
            result = U32Range{static_cast<u32>(min), static_cast<u32>(max)};
        }
        break;
    }
    case IR::Opcode::IMul32: {
        const auto lhs = range(0);
        const auto rhs = range(1);
        if (!lhs || !rhs) {
            break;
        }
        const u64 min = static_cast<u64>(lhs->min) * rhs->min;
        const u64 max = static_cast<u64>(lhs->max) * rhs->max;
        if (max <= std::numeric_limits<u32>::max()) {
            result = U32Range{static_cast<u32>(min), static_cast<u32>(max)};
        }
        break;
    }
    case IR::Opcode::BitwiseAnd32: {
        const auto lhs = range(0);
        const auto rhs = range(1);
        if (!lhs || !rhs) {
            break;
        }
        if (lhs->min == lhs->max && rhs->min == rhs->max) {
            const u32 value = lhs->min & rhs->min;
            result = U32Range{value, value};
        } else if (lhs->min == lhs->max) {
            result = U32Range{0, lhs->min & PossibleBits(rhs->max)};
        } else if (rhs->min == rhs->max) {
            result = U32Range{0, rhs->min & PossibleBits(lhs->max)};
        }
        break;
    }
    default:
        break;
    }

    visiting.erase(inst);
    return result;
}

static bool IsDivergentCondition(const IR::U1& cond, const std::array<u32, 3>& workgroup_size) {
    if (cond.IsImmediate()) {
        return false;
    }

    InstSet visited;
    std::queue<IR::Inst*> pending;
    pending.push(cond.InstRecursive());
    while (!pending.empty()) {
        IR::Inst* const inst = pending.front();
        pending.pop();
        if (!visited.emplace(inst).second) {
            continue;
        }

        InstSet visiting;
        const auto range = GetWorkgroupU32Range(IR::Value{inst}, workgroup_size, visiting);
        if (range && range->min == range->max) {
            continue;
        }
        if (inst->GetOpcode() == IR::Opcode::GetAttributeU32 &&
            inst->Arg(0).Attribute() == IR::Attribute::LocalInvocationId) {
            if (!inst->Arg(1).IsImmediate()) {
                return true;
            }
            const u32 component = inst->Arg(1).U32();
            if (component >= workgroup_size.size() || workgroup_size[component] > 1) {
                return true;
            }
        }
        for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
            const IR::Value value = inst->Arg(arg);
            if (!value.IsImmediate()) {
                pending.push(value.InstRecursive());
            }
        }
    }
    return false;
}

static void EmitBarrierAtMerge(IR::Block* merge) {
    auto insert_point = std::ranges::find_if_not(merge->Instructions(), IR::IsPhi);
    IR::IREmitter ir{*merge, insert_point};
    ir.Barrier();
}

static NodeSet FindDivergentLoops(const IR::Program& program,
                                  const std::array<u32, 3>& workgroup_size) {
    struct LoopScope {
        IR::Block* merge;
        bool divergent;
    };

    using Type = IR::AbstractSyntaxNode::Type;
    std::vector<LoopScope> loop_stack;
    std::vector<bool> if_divergence_stack;
    u32 if_divergence_depth{};
    NodeSet divergent_loops;
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        if (node.type == Type::If) {
            const bool divergent = IsDivergentCondition(node.data.if_node.cond, workgroup_size);
            if_divergence_stack.push_back(divergent);
            if_divergence_depth += divergent;
            continue;
        }
        if (node.type == Type::EndIf) {
            ASSERT(!if_divergence_stack.empty());
            if_divergence_depth -= if_divergence_stack.back();
            if_divergence_stack.pop_back();
            continue;
        }
        if (node.type == Type::Loop) {
            loop_stack.push_back({
                .merge = node.data.loop.merge,
                .divergent = false,
            });
            continue;
        }
        if (node.type == Type::Break &&
            (if_divergence_depth != 0 ||
             IsDivergentCondition(node.data.break_node.cond, workgroup_size))) {
            for (auto scope = loop_stack.rbegin(); scope != loop_stack.rend(); ++scope) {
                if (scope->merge == node.data.break_node.merge) {
                    scope->divergent = true;
                    break;
                }
            }
            continue;
        }
        if (node.type != Type::Repeat || loop_stack.empty()) {
            continue;
        }
        LoopScope scope = loop_stack.back();
        loop_stack.pop_back();
        if (if_divergence_depth != 0 ||
            IsDivergentCondition(node.data.repeat.cond, workgroup_size)) {
            scope.divergent = true;
        }
        if (scope.divergent) {
            divergent_loops.emplace(scope.merge);
        }
    }
    return divergent_loops;
}

// Inserts a barrier after divergent conditional blocks to avoid undefined
// behavior when some threads write and others read from shared memory.
static void EmitBarrierInMergeBlock(const IR::AbstractSyntaxNode::Data& data,
                                    const std::array<u32, 3>& workgroup_size,
                                    NodeSet& divergence_end, u32& divergence_depth) {
    const IR::U1 cond = data.if_node.cond;
    if (IsDivergentCondition(cond, workgroup_size)) {
        if (divergence_depth == 0) {
            EmitBarrierAtMerge(data.if_node.merge);
        }
        ++divergence_depth;
        divergence_end.emplace(data.if_node.merge);
    }
}

void SharedMemoryBarrierPass(IR::Program& program, const RuntimeInfo& runtime_info,
                             const Profile& profile) {
    if (program.info.stage != Stage::Compute) {
        return;
    }
    const auto& cs_info = runtime_info.cs_info;
    const u32 shared_memory_size = cs_info.shared_memory_size;
    const u32 threadgroup_size =
        cs_info.workgroup_size[0] * cs_info.workgroup_size[1] * cs_info.workgroup_size[2];
    // Guest shaders can use wave-synchronous LDS accesses inside any workgroup size. Preserve
    // their visibility on hosts whose subgroups do not provide the same LDS ordering; the pass
    // already avoids inserting barriers in divergent control flow.
    if (shared_memory_size == 0 || threadgroup_size == 0 || !profile.needs_lds_barriers) {
        return;
    }
    using Type = IR::AbstractSyntaxNode::Type;
    u32 divergence_depth{};
    NodeSet divergence_end;
    const NodeSet divergent_loops = FindDivergentLoops(program, cs_info.workgroup_size);
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        if (node.type == Type::EndIf) {
            if (divergence_end.contains(node.data.end_if.merge)) {
                --divergence_depth;
            }
            continue;
        }
        if (node.type == Type::Loop) {
            if (divergent_loops.contains(node.data.loop.merge)) {
                if (divergence_depth == 0) {
                    EmitBarrierAtMerge(node.data.loop.merge);
                }
                ++divergence_depth;
            }
            continue;
        }
        if (node.type == Type::Repeat) {
            if (divergent_loops.contains(node.data.repeat.merge)) {
                --divergence_depth;
            }
            continue;
        }
        // Check if branch depth is zero, we don't want to insert barrier in potentially divergent
        // code.
        if (node.type == Type::If) {
            EmitBarrierInMergeBlock(node.data, cs_info.workgroup_size, divergence_end,
                                    divergence_depth);
            continue;
        }
        if (node.type == Type::Block && divergence_depth == 0) {
            EmitBarrierInBlock(node.data.block);
        }
    }
}

} // namespace Shader::Optimization
