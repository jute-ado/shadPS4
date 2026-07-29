// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "common/alignment.h"
#include "shader_recompiler/ir/breadth_first_search.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Optimization {
namespace {

constexpr u32 GcnWaveSize = 64;
constexpr u32 DwordSize = sizeof(u32);

using BlockSet = std::unordered_set<IR::Block*>;
using ReadBatch = std::vector<IR::Inst*>;

BlockSet FindConvergedBlocks(const IR::Program& program) {
    using Type = IR::AbstractSyntaxNode::Type;

    BlockSet blocks;
    u32 conditional_depth{};
    u32 loop_depth{};
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        switch (node.type) {
        case Type::Block:
            if (conditional_depth == 0 && loop_depth == 0) {
                blocks.emplace(node.data.block);
            }
            break;
        case Type::If:
            ++conditional_depth;
            break;
        case Type::EndIf:
            if (conditional_depth != 0) {
                --conditional_depth;
            }
            break;
        case Type::Loop:
            ++loop_depth;
            break;
        case Type::Repeat:
            if (loop_depth != 0) {
                --loop_depth;
            }
            break;
        default:
            break;
        }
    }
    return blocks;
}

bool IsFixedReadLane(const IR::Inst& inst) {
    return inst.GetOpcode() == IR::Opcode::ReadLane && inst.HasUses() &&
           inst.Arg(1).IsImmediate() && inst.Arg(1).U32() < GcnWaveSize;
}

bool DependsOnBatch(const IR::Value& value, const ReadBatch& batch) {
    const auto dependency =
        IR::BreadthFirstSearch(value, [&batch](IR::Inst* inst) -> std::optional<bool> {
            if (std::ranges::find(batch, inst) != batch.end()) {
                return true;
            }
            return std::nullopt;
        });
    return dependency.value_or(false);
}

std::vector<ReadBatch> FindBatches(IR::Program& program, const BlockSet& converged_blocks) {
    std::vector<ReadBatch> batches;
    for (IR::Block* const block : program.blocks) {
        if (!converged_blocks.contains(block)) {
            continue;
        }

        ReadBatch batch;
        const auto flush_batch = [&] {
            if (!batch.empty()) {
                batches.emplace_back(std::move(batch));
                batch.clear();
            }
        };
        for (IR::Inst& inst : block->Instructions()) {
            if (!IsFixedReadLane(inst)) {
                // SSA register setters remain in the list until dead-code elimination. They have
                // no observable effect and should not prevent adjacent guest lane reads from
                // sharing one exchange.
                if (inst.HasUses() || inst.MayHaveSideEffects()) {
                    flush_batch();
                }
                continue;
            }
            if (DependsOnBatch(inst.Arg(0), batch)) {
                flush_batch();
            }
            batch.push_back(&inst);
        }
        flush_batch();
    }
    return batches;
}

IR::U32 LocalInvocationIndex(IR::IREmitter& ir, const ComputeRuntimeInfo& cs_info) {
    const IR::U32 x = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U32 y = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1);
    const IR::U32 z = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 2);
    const IR::U32 xy = ir.IAdd(x, ir.IMul(y, ir.Imm32(cs_info.workgroup_size[0])));
    return ir.IAdd(xy, ir.IMul(z, ir.Imm32(cs_info.workgroup_size[0] * cs_info.workgroup_size[1])));
}

void LowerBatch(const ReadBatch& batch, size_t begin, size_t count, u32 scratch_base, u32 bank_size,
                const ComputeRuntimeInfo& cs_info) {
    IR::Block* const block = batch[begin]->GetParent();
    const auto insert_point = IR::Block::InstructionList::s_iterator_to(*batch[begin]);
    IR::IREmitter ir{*block, insert_point};

    const IR::U32 invocation_index = LocalInvocationIndex(ir, cs_info);
    const IR::U32 invocation_byte_offset = ir.IMul(invocation_index, ir.Imm32(DwordSize));
    const IR::U32 wave_index = ir.BitwiseAnd(invocation_index, ir.Imm32(~(GcnWaveSize - 1U)));
    const IR::U32 wave_byte_offset = ir.IMul(wave_index, ir.Imm32(DwordSize));

    for (size_t index = 0; index < count; ++index) {
        IR::Inst* const read = batch[begin + index];
        const u32 bank_base = scratch_base + static_cast<u32>(index) * bank_size;
        const IR::U32 write_offset = ir.IAdd(ir.Imm32(bank_base), invocation_byte_offset);
        ir.WriteShared(32, read->Arg(0), write_offset);
    }
    ir.Barrier();

    std::vector<IR::Value> replacements;
    replacements.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        IR::Inst* const read = batch[begin + index];
        const u32 lane_byte_offset = read->Arg(1).U32() * DwordSize;
        const u32 bank_base = scratch_base + static_cast<u32>(index) * bank_size;
        const IR::U32 read_offset =
            ir.IAdd(ir.Imm32(bank_base + lane_byte_offset), wave_byte_offset);
        replacements.emplace_back(ir.LoadShared(32, false, read_offset));
    }
    ir.Barrier();

    for (size_t index = 0; index < count; ++index) {
        batch[begin + index]->ReplaceUsesWithAndRemove(replacements[index]);
    }
}

} // Anonymous namespace

void ReadLaneWorkgroupPass(IR::Program& program, RuntimeInfo& runtime_info,
                           const Profile& profile) {
    // A fixed GCN lane number addresses the whole 64-lane guest wave. Native SPIR-V subgroup
    // broadcast cannot preserve that meaning when the host splits the wave into smaller
    // subgroups: lane 0 in the second host subgroup is guest lane 32, for example. At converged
    // points in a one-wave compute workgroup, shared memory provides an exact cross-subgroup
    // exchange without relying on an out-of-range subgroup lane.
    if (program.info.stage != Stage::Compute || profile.subgroup_size == 0 ||
        profile.subgroup_size >= GcnWaveSize || profile.supports_compute_subgroup_size_64) {
        return;
    }

    auto& cs_info = runtime_info.cs_info;
    const u64 thread_count = static_cast<u64>(cs_info.workgroup_size[0]) *
                             cs_info.workgroup_size[1] * cs_info.workgroup_size[2];
    // A workgroup barrier can safely emulate a wave operation only when the workgroup is exactly
    // one guest wave. Multiple guest waves may take independent scalar control-flow paths.
    if (thread_count != GcnWaveSize) {
        return;
    }

    const BlockSet converged_blocks = FindConvergedBlocks(program);
    const std::vector<ReadBatch> batches = FindBatches(program, converged_blocks);
    if (batches.empty()) {
        return;
    }

    const u32 scratch_base = Common::AlignUp(cs_info.shared_memory_size, DwordSize);
    const u64 bank_size = thread_count * DwordSize;
    if (scratch_base >= profile.max_shared_memory_size ||
        bank_size > profile.max_shared_memory_size - scratch_base) {
        return;
    }

    size_t largest_batch{};
    for (const ReadBatch& batch : batches) {
        largest_batch = std::max(largest_batch, batch.size());
    }
    const size_t available_banks =
        static_cast<size_t>((profile.max_shared_memory_size - scratch_base) / bank_size);
    const size_t bank_count = std::min(largest_batch, available_banks);
    if (bank_count == 0) {
        return;
    }

    const u32 bank_size_u32 = static_cast<u32>(bank_size);
    cs_info.shared_memory_size = scratch_base + static_cast<u32>(bank_count) * bank_size_u32;
    for (const ReadBatch& batch : batches) {
        for (size_t begin = 0; begin < batch.size(); begin += bank_count) {
            const size_t count = std::min(bank_count, batch.size() - begin);
            LowerBatch(batch, begin, count, scratch_base, bank_size_u32, cs_info);
        }
    }
}

} // namespace Shader::Optimization
