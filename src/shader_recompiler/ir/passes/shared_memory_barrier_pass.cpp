// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_set>
#include <vector>
#include "common/alignment.h"
#include "shader_recompiler/ir/breadth_first_search.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"

namespace Shader::Optimization {

namespace {

constexpr u32 GcnWaveSize = 64;
constexpr u32 Wave64BallotScratchBytes = 2 * sizeof(u32);

IR::U32 LocalInvocationIndex(IR::IREmitter& ir, const ComputeRuntimeInfo& cs_info) {
    const IR::U32 x = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U32 y = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1);
    const IR::U32 z = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 2);
    const IR::U32 xy = ir.IAdd(x, ir.IMul(y, ir.Imm32(cs_info.workgroup_size[0])));
    return ir.IAdd(xy,
                   ir.IMul(z, ir.Imm32(cs_info.workgroup_size[0] * cs_info.workgroup_size[1])));
}

std::pair<IR::U32, IR::U32> EmitWave64Ballot(IR::IREmitter& ir, const IR::U1& bit,
                                             u32 scratch_base,
                                             const ComputeRuntimeInfo& cs_info) {
    const IR::U32 lane = LocalInvocationIndex(ir, cs_info);
    const IR::U32 lane_in_word = ir.BitwiseAnd(lane, ir.Imm32(31U));
    const IR::U32 lane_word = ir.ShiftRightLogical(lane, ir.Imm32(5U));
    const IR::U32 word_offset =
        ir.IAdd(ir.Imm32(scratch_base), ir.ShiftLeftLogical(lane_word, ir.Imm32(2U)));

    // Every invocation resets its own word atomically, then all 64 invocations publish their
    // predicate bit. This is deliberately workgroup-wide: on a wave32 host, a native ballot can
    // never produce the high half of a PS4 wave64 mask. The barriers are unconditional so both
    // host subgroups participate even when the scalar fallback is ultimately selected.
    static_cast<void>(ir.SharedAtomicAnd(word_offset, ir.Imm32(0U), false));
    ir.Barrier();
    const IR::U32 bit_value{ir.Select(
        bit, ir.ShiftLeftLogical(ir.Imm32(1U), lane_in_word), ir.Imm32(0U))};
    static_cast<void>(ir.SharedAtomicOr(word_offset, bit_value, false));
    ir.Barrier();
    const IR::U32 low{ir.LoadShared(32, false, ir.Imm32(scratch_base))};
    const IR::U32 high{ir.LoadShared(
        32, false, ir.Imm32(static_cast<u32>(scratch_base + sizeof(u32))))};
    // Prevent a faster host subgroup from resetting the shared words for the next guest ballot
    // before the other subgroup has consumed this one.
    ir.Barrier();
    return {low, high};
}

} // Anonymous namespace

void Wave64MaskLoweringPass(IR::Program& program, RuntimeInfo& runtime_info,
                            const Profile& profile) {
    if (program.info.stage != Stage::Compute || profile.subgroup_size == 0 ||
        profile.subgroup_size >= GcnWaveSize) {
        return;
    }

    auto& cs_info = runtime_info.cs_info;
    const u64 thread_count = static_cast<u64>(cs_info.workgroup_size[0]) *
                             cs_info.workgroup_size[1] * cs_info.workgroup_size[2];
    if (thread_count != GcnWaveSize) {
        return;
    }

    std::vector<IR::Inst*> lane_ids;
    std::vector<IR::Inst*> masks;
    for (IR::Block* block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::LaneId) {
                lane_ids.push_back(&inst);
            } else if (inst.GetOpcode() == IR::Opcode::Ballot ||
                       inst.GetOpcode() == IR::Opcode::MaterializeScalarMaskWord) {
                masks.push_back(&inst);
            }
        }
    }
    if (lane_ids.empty() && masks.empty()) {
        return;
    }

    const u32 scratch_base = Common::AlignUp(cs_info.shared_memory_size, sizeof(u32));
    if (!masks.empty() &&
        (scratch_base > profile.max_shared_memory_size ||
         Wave64BallotScratchBytes > profile.max_shared_memory_size - scratch_base)) {
        return;
    }

    for (IR::Inst* lane_id : lane_ids) {
        IR::IREmitter ir{*lane_id->GetParent(),
                         IR::Block::InstructionList::s_iterator_to(*lane_id)};
        lane_id->ReplaceUsesWithAndRemove(LocalInvocationIndex(ir, cs_info));
    }
    if (masks.empty()) {
        return;
    }

    cs_info.shared_memory_size = scratch_base + Wave64BallotScratchBytes;
    for (IR::Inst* mask : masks) {
        IR::IREmitter ir{*mask->GetParent(), IR::Block::InstructionList::s_iterator_to(*mask)};
        const IR::U1 bit{mask->GetOpcode() == IR::Opcode::Ballot ? mask->Arg(0) : mask->Arg(1)};
        const auto [low, high] = EmitWave64Ballot(ir, bit, scratch_base, cs_info);
        if (mask->GetOpcode() == IR::Opcode::Ballot) {
            mask->ReplaceUsesWithAndRemove(
                ir.CompositeConstruct(low, high, ir.Imm32(0U), ir.Imm32(0U)));
            continue;
        }
        const u32 word = mask->Arg(3).U32();
        const IR::U32 ballot_word = word == 0 ? low : high;
        mask->ReplaceUsesWithAndRemove(
            ir.Select(IR::U1{mask->Arg(0)}, ballot_word, IR::U32{mask->Arg(2)}));
    }
}

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

// Inserts a barrier after divergent conditional blocks to avoid undefined
// behavior when some threads write and others read from shared memory.
static void EmitBarrierInMergeBlock(const IR::AbstractSyntaxNode::Data& data,
                                    NodeSet& divergence_end, u32& divergence_depth) {
    const IR::U1 cond = data.if_node.cond;
    const auto is_divergent_cond =
        IR::BreadthFirstSearch(cond, [](IR::Inst* inst) -> std::optional<bool> {
            if (inst->GetOpcode() == IR::Opcode::LaneId) {
                return true;
            }
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
    // A guest compiler may omit explicit barriers when the entire threadgroup fits in one
    // wave64. A narrower host subgroup does not preserve that implicit LDS ordering, so restore
    // barriers for every one-wave guest threadgroup, including groups smaller than 64 threads.
    if (shared_memory_size == 0 || threadgroup_size > GcnWaveSize ||
        !profile.needs_lds_barriers) {
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
