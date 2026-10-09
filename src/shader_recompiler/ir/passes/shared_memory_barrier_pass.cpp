// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_set>
#include <optional>
#include "shader_recompiler/ir/breadth_first_search.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/shared_memory_barrier_policy.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"

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
    std::optional<SharedMemoryAccess> previous_access;
    for (IR::Inst& inst : block->Instructions()) {
        const auto current_access = IsLoadShared(inst)   ? std::optional{SharedMemoryAccess::Load}
                                    : IsWriteShared(inst) ? std::optional{SharedMemoryAccess::Write}
                                                          : std::nullopt;
        if (!current_access) {
            continue;
        }
        if (previous_access && NeedsSharedMemoryBarrier(*previous_access, *current_access)) {
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            ir.Barrier();
        }
        previous_access = current_access;
    }
    if (previous_access) {
        IR::IREmitter ir{*block, --block->end()};
        ir.Barrier();
    }
}

using NodeSet = std::unordered_set<const IR::Block*>;

// Inserts a barrier after divergent conditional blocks to avoid undefined
// behavior when some threads write and others read from shared memory.
static void EmitBarrierInMergeBlock(const IR::AbstractSyntaxNode::Data& data,
                                    NodeSet& divergence_end, u32& divergence_depth) {
    const IR::U1 cond = data.if_node.cond;
    const auto is_divergent_cond =
        IR::BreadthFirstSearch(cond, [](IR::Inst* inst) -> std::optional<bool> {
            if (inst->GetOpcode() == IR::Opcode::GetAttributeU32 &&
                inst->Arg(0).Attribute() == IR::Attribute::LocalInvocationId) {
                return true;
            }
            return std::nullopt;
        });
    if (is_divergent_cond) {
        if (divergence_depth == 0) {
            IR::Block* const merge = data.if_node.merge;
            auto insert_point = std::ranges::find_if_not(merge->Instructions(), IR::IsPhi);
            IR::IREmitter ir{*merge, insert_point};
            ir.Barrier();
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
    // Guest wave64 regions execute in lockstep on PS4 even when they are part of a larger
    // workgroup. Explicitly preserve their LDS ordering on narrower host subgroups. Barriers are
    // inserted only in converged blocks, so a workgroup-wide host barrier is a safe stronger
    // ordering for multi-wave workgroups as well.
    if (shared_memory_size == 0 ||
        !NeedsSharedMemoryBarrierEmulation(threadgroup_size, profile.needs_lds_barriers)) {
        return;
    }
    using Type = IR::AbstractSyntaxNode::Type;
    u32 divergence_depth{};
    NodeSet divergence_end;
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        if (node.type == Type::EndIf) {
            if (divergence_end.contains(node.data.end_if.merge)) {
                --divergence_depth;
            }
            continue;
        }
        // Check if branch depth is zero, we don't want to insert barrier in potentially divergent
        // code.
        if (node.type == Type::If) {
            EmitBarrierInMergeBlock(node.data, divergence_end, divergence_depth);
            continue;
        }
        if (node.type == Type::Block && divergence_depth == 0) {
            EmitBarrierInBlock(node.data.block);
        }
    }
}

} // namespace Shader::Optimization
