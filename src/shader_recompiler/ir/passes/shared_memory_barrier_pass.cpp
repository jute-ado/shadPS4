// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <optional>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "common/alignment.h"
#include "common/logging/log.h"
#include "shader_recompiler/ir/breadth_first_search.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"

namespace Shader::Optimization {

namespace {

constexpr u32 GcnWaveSize = 64;
constexpr u32 Wave64ExchangeScratchBytes = GcnWaveSize * sizeof(u32);
constexpr u32 Wave64GroupAnyScratchBytes = 2 * sizeof(u32);
constexpr u32 Wave64ReadFirstActiveScratchBytes = 4 * sizeof(u32);
constexpr u32 Wave64ControlScratchBytes = 6 * sizeof(u32);

struct Wave64ControlSyncPoint {
    IR::Block* block{};
    IR::U1 condition;
    bool invert{};
};

struct Wave64ControlledOperation {
    IR::Block* init_block{};
    std::vector<Wave64ControlSyncPoint> sync_points;
    IR::Inst* operation{};
    bool sync_after_init{};
};

struct Wave64ControlFrame {
    enum class Type {
        If,
        Loop,
    };

    Type type{};
    IR::Block* entry_block{};
    IR::Block* merge{};
    IR::U1 condition;
    IR::Block* break_block{};
    IR::U1 break_condition;
    size_t break_count{};
    bool unsafe_exit{};
};

IR::U32 LocalInvocationIndex(IR::IREmitter& ir, const ComputeRuntimeInfo& cs_info) {
    const IR::U32 x = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U32 y = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1);
    const IR::U32 z = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 2);
    const IR::U32 xy = ir.IAdd(x, ir.IMul(y, ir.Imm32(cs_info.workgroup_size[0])));
    return ir.IAdd(xy,
                   ir.IMul(z, ir.Imm32(cs_info.workgroup_size[0] * cs_info.workgroup_size[1])));
}

bool IsUnsupportedWaveOperation(const IR::Inst& inst, const Profile& profile) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::WarpId:
    case IR::Opcode::QuadShuffle:
    case IR::Opcode::DataAppend:
    case IR::Opcode::DataConsume:
        return true;
    case IR::Opcode::ShuffleXor:
        // An XOR confined to the host subgroup selects the same guest lane. Cross-half and
        // dynamic masks need a workgroup exchange and cannot remain native.
        return !inst.Arg(1).IsImmediate() || inst.Arg(1).U32() >= profile.subgroup_size;
    default:
        return false;
    }
}

std::unordered_set<const IR::Block*> WorkgroupExchangeBlocks(const IR::Program& program) {
    using Type = IR::AbstractSyntaxNode::Type;
    std::unordered_set<const IR::Block*> blocks;
    std::unordered_map<const IR::Block*, u32> divergent_break_merges;
    u32 divergent_depth{};
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        switch (node.type) {
        case Type::Block:
            if (const auto it = divergent_break_merges.find(node.data.block);
                it != divergent_break_merges.end()) {
                ASSERT(divergent_depth >= it->second);
                divergent_depth -= it->second;
                divergent_break_merges.erase(it);
            }
            if (divergent_depth == 0) {
                blocks.emplace(node.data.block);
            }
            break;
        case Type::If:
            if (!node.data.if_node.is_guest_wave_uniform) {
                ++divergent_depth;
            }
            break;
        case Type::Loop:
            if (!node.data.loop.is_guest_wave_uniform) {
                ++divergent_depth;
            }
            break;
        case Type::EndIf:
            if (!node.data.end_if.is_guest_wave_uniform) {
                ASSERT(divergent_depth > 0);
                --divergent_depth;
            }
            break;
        case Type::Repeat:
            if (!node.data.repeat.is_guest_wave_uniform) {
                ASSERT(divergent_depth > 0);
                --divergent_depth;
                // A divergent loop header is a back-edge target that some invocations may stop
                // reaching. Guest scalar loops remain workgroup-uniform by construction.
                blocks.erase(node.data.repeat.loop_header);
            }
            break;
        case Type::Break:
            if (!node.data.break_node.is_guest_wave_uniform) {
                ++divergent_depth;
                ++divergent_break_merges[node.data.break_node.merge];
            }
            break;
        default:
            break;
        }
    }
    ASSERT(divergent_depth == 0);
    ASSERT(divergent_break_merges.empty());
    return blocks;
}

void LowerSingleSubgroupPartialWaveOperations(IR::Program& program,
                                              const ComputeRuntimeInfo& cs_info,
                                              const Profile& profile, u64 thread_count) {
    if (thread_count == 0 || thread_count >= GcnWaveSize ||
        thread_count > profile.subgroup_size) {
        return;
    }

    std::vector<IR::Inst*> inactive_reads;
    std::vector<IR::Inst*> writes;
    for (IR::Block* block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::ReadLane && inst.Arg(1).IsImmediate()) {
                const u32 source_lane = inst.Arg(1).U32() & (GcnWaveSize - 1);
                if (source_lane >= thread_count) {
                    inactive_reads.push_back(&inst);
                }
            } else if (inst.GetOpcode() == IR::Opcode::WriteLane) {
                writes.push_back(&inst);
            }
        }
    }

    for (IR::Inst* read_lane : inactive_reads) {
        // GCN wave64 lanes which do not belong to a partial final wave are inactive and their
        // vector-register contents are undefined. Do not pass an inactive guest lane index to a
        // smaller Vulkan subgroup: an index equal to SubgroupSize is itself out of range and can
        // reach undefined driver behavior. Resolve the guest undefined value deterministically.
        read_lane->ReplaceUsesWithAndRemove(IR::Value{u32{0}});
    }
    for (IR::Inst* write_lane : writes) {
        const IR::Value target = write_lane->Arg(2);
        if (target.IsImmediate() && (target.U32() & (GcnWaveSize - 1)) >= thread_count) {
            write_lane->ReplaceUsesWithAndRemove(write_lane->Arg(0));
            continue;
        }

        IR::IREmitter ir{*write_lane->GetParent(),
                         IR::Block::InstructionList::s_iterator_to(*write_lane)};
        const IR::U32 local_lane = LocalInvocationIndex(ir, cs_info);
        const IR::U32 target_lane =
            ir.BitwiseAnd(IR::U32{target}, ir.Imm32(GcnWaveSize - 1));
        const IR::U32 value{ir.Select(ir.IEqual(local_lane, target_lane),
                                     write_lane->Arg(1), write_lane->Arg(0))};
        write_lane->ReplaceUsesWithAndRemove(value);
    }
    if (!inactive_reads.empty()) {
        LOG_INFO(Render_Recompiler, "Resolved {} inactive ReadLane operation(s) in a {}-thread "
                                    "partial guest wave",
                 inactive_reads.size(), thread_count);
    }
    if (!writes.empty()) {
        LOG_INFO(Render_Recompiler, "Lowered {} WriteLane operation(s) in a {}-thread partial "
                                    "guest wave",
                 writes.size(), thread_count);
    }
}

std::optional<std::vector<Wave64ControlledOperation>> FindWave64ControlledOperations(
    IR::Program& program, IR::Opcode opcode, bool include_converged) {
    using Type = IR::AbstractSyntaxNode::Type;

    struct LoopExitInfo {
        size_t count{};
        IR::Block* block{};
        IR::U1 condition;
    };
    std::unordered_map<IR::Block*, LoopExitInfo> loop_exits;
    std::unordered_set<IR::Block*> unsafe_control_regions;
    std::vector<std::pair<Type, IR::Block*>> scan_frames;
    IR::Block* scan_block{};
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        if (node.type == Type::Block) {
            scan_block = node.data.block;
            if (std::ranges::any_of(scan_block->Instructions(), [](const IR::Inst& inst) {
                    return inst.GetOpcode() == IR::Opcode::Discard ||
                           inst.GetOpcode() == IR::Opcode::DiscardCond;
                })) {
                for (const auto& [type, merge] : scan_frames) {
                    unsafe_control_regions.insert(merge);
                }
            }
        } else if (node.type == Type::If) {
            scan_frames.emplace_back(Type::If, node.data.if_node.merge);
        } else if (node.type == Type::EndIf) {
            if (scan_frames.empty() || scan_frames.back().first != Type::If ||
                scan_frames.back().second != node.data.end_if.merge) {
                return std::nullopt;
            }
            scan_frames.pop_back();
        } else if (node.type == Type::Loop) {
            scan_frames.emplace_back(Type::Loop, node.data.loop.merge);
        } else if (node.type == Type::Break) {
            LoopExitInfo& exit = loop_exits[node.data.break_node.merge];
            ++exit.count;
            if (exit.count == 1) {
                exit.block = scan_block;
                exit.condition = node.data.break_node.cond;
            }
        } else if (node.type == Type::Repeat) {
            if (scan_frames.empty() || scan_frames.back().first != Type::Loop ||
                scan_frames.back().second != node.data.repeat.merge) {
                return std::nullopt;
            }
            scan_frames.pop_back();
        } else if (node.type == Type::Return || node.type == Type::Unreachable) {
            for (const auto& [type, merge] : scan_frames) {
                unsafe_control_regions.insert(merge);
            }
        }
    }

    std::vector<Wave64ControlledOperation> patterns;
    std::vector<Wave64ControlFrame> frames;
    IR::Block* current_block{};

    const auto mark_unsafe_exit = [&] {
        for (Wave64ControlFrame& frame : frames) {
            frame.unsafe_exit = true;
        }
    };

    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        switch (node.type) {
        case Type::Block: {
            current_block = node.data.block;
            if (std::ranges::any_of(current_block->Instructions(), [](const IR::Inst& inst) {
                    return inst.GetOpcode() == IR::Opcode::Discard ||
                           inst.GetOpcode() == IR::Opcode::DiscardCond;
                })) {
                mark_unsafe_exit();
            }
            for (IR::Inst& inst : current_block->Instructions()) {
                if (inst.GetOpcode() != opcode || !inst.HasUses()) {
                    continue;
                }
                // A converged ReadFirstLane uses the ordinary workgroup exchange. This protocol
                // is specifically for reads whose participating lanes are controlled by
                // structured guest EXEC flow.
                if (frames.empty()) {
                    if (include_converged) {
                        patterns.push_back({
                            .init_block = current_block,
                            .operation = &inst,
                            .sync_after_init = true,
                        });
                    }
                    continue;
                }

                // The control-state initialization must execute exactly once before either
                // host half starts publishing generations. A top-level loop header is reached
                // again on every iteration, so using it as the initializer can reset a peer's
                // generation while that peer is still waiting. Decline until a true converged
                // preheader can be represented for this structured shape.
                if (frames.front().type == Wave64ControlFrame::Type::Loop) {
                    return std::nullopt;
                }

                Wave64ControlledOperation pattern{
                    .init_block = frames.front().entry_block,
                    .operation = &inst,
                };
                for (const Wave64ControlFrame& frame : frames) {
                    if (frame.unsafe_exit || frame.entry_block == nullptr) {
                        return std::nullopt;
                    }
                    if (frame.type == Wave64ControlFrame::Type::If) {
                        pattern.sync_points.push_back({
                            .block = frame.entry_block,
                            .condition = frame.condition,
                        });
                        continue;
                    }
                    // A lane-retiring scalar loop must publish the surviving lanes before the
                    // semantic read. Multiple exits cannot be ordered safely by this protocol.
                    if (frame.break_count != 1 || frame.break_block == nullptr) {
                        return std::nullopt;
                    }
                    pattern.sync_points.push_back({
                        .block = frame.break_block,
                        .condition = frame.break_condition,
                        .invert = true,
                    });
                }
                patterns.push_back(std::move(pattern));
            }
            break;
        }
        case Type::If:
            if (current_block == nullptr) {
                return std::nullopt;
            }
            frames.push_back({
                .type = Wave64ControlFrame::Type::If,
                .entry_block = current_block,
                .merge = node.data.if_node.merge,
                .condition = node.data.if_node.cond,
                .unsafe_exit = unsafe_control_regions.contains(node.data.if_node.merge),
            });
            break;
        case Type::EndIf:
            if (frames.empty() || frames.back().type != Wave64ControlFrame::Type::If ||
                frames.back().merge != node.data.end_if.merge) {
                return std::nullopt;
            }
            frames.pop_back();
            break;
        case Type::Loop:
            if (current_block == nullptr) {
                return std::nullopt;
            }
            frames.push_back({
                .type = Wave64ControlFrame::Type::Loop,
                .entry_block = current_block,
                .merge = node.data.loop.merge,
                .unsafe_exit = unsafe_control_regions.contains(node.data.loop.merge),
            });
            if (const auto exit = loop_exits.find(node.data.loop.merge);
                exit != loop_exits.end()) {
                frames.back().break_block = exit->second.block;
                frames.back().break_condition = exit->second.condition;
                frames.back().break_count = exit->second.count;
            }
            break;
        case Type::Break:
            break;
        case Type::Repeat:
            if (frames.empty() || frames.back().type != Wave64ControlFrame::Type::Loop ||
                frames.back().merge != node.data.repeat.merge) {
                return std::nullopt;
            }
            frames.pop_back();
            break;
        case Type::Return:
        case Type::Unreachable:
            mark_unsafe_exit();
            break;
        }
    }
    return patterns;
}

void InitializeWave64ControlState(const Wave64ControlledOperation& pattern,
                                  u32 pattern_scratch,
                                  const ComputeRuntimeInfo& cs_info) {
    {
        IR::IREmitter ir{*pattern.init_block};
        const IR::U32 invocation = LocalInvocationIndex(ir, cs_info);
        ir.Wave64LaneRetirementInit(ir.Imm1(true), invocation,
                                    ir.Imm32(pattern_scratch));
        // Initialization is the only workgroup-wide rendezvous. It is inserted before the
        // outermost structured control node, where the whole one-wave workgroup is active.
        if (pattern.sync_after_init) {
            ir.Wave64LaneRetirementSync(ir.Imm1(true), invocation,
                                        ir.Imm32(pattern_scratch));
        }
    }
    for (const Wave64ControlSyncPoint& sync : pattern.sync_points) {
        IR::IREmitter ir{*sync.block};
        const IR::U32 invocation = LocalInvocationIndex(ir, cs_info);
        const IR::U1 continues =
            sync.invert ? ir.LogicalNot(sync.condition) : sync.condition;
        ir.Wave64LaneRetirementSync(continues, invocation, ir.Imm32(pattern_scratch));
    }
}

void LowerWave64ReadFirstLanePatterns(
    const std::vector<Wave64ControlledOperation>& patterns, u32 scratch_base,
    const ComputeRuntimeInfo& cs_info) {
    for (size_t index = 0; index < patterns.size(); ++index) {
        const Wave64ControlledOperation& pattern = patterns[index];
        const u32 pattern_scratch =
            scratch_base + static_cast<u32>(index) * Wave64ControlScratchBytes;
        InitializeWave64ControlState(pattern, pattern_scratch, cs_info);
        {
            const auto insert_point =
                IR::Block::InstructionList::s_iterator_to(*pattern.operation);
            IR::IREmitter ir{*pattern.operation->GetParent(), insert_point};
            const IR::U32 invocation = LocalInvocationIndex(ir, cs_info);
            const IR::U32 replacement = ir.Wave64ReadFirstLane(
                IR::U32{pattern.operation->Arg(0)}, IR::U1{pattern.operation->Arg(1)},
                invocation, ir.Imm32(pattern_scratch));
            pattern.operation->ReplaceUsesWithAndRemove(replacement);
        }
    }
}

void LowerConvergedReadFirstActiveLanes(const std::vector<IR::Inst*>& operations,
                                        u32 scratch_base,
                                        const ComputeRuntimeInfo& cs_info) {
    for (IR::Inst* operation : operations) {
        IR::IREmitter ir{*operation->GetParent(),
                         IR::Block::InstructionList::s_iterator_to(*operation)};
        const IR::U32 invocation = LocalInvocationIndex(ir, cs_info);
        const IR::U32 replacement = ir.Wave64ReadFirstActiveLane(
            IR::U32{operation->Arg(0)}, IR::U1{operation->Arg(1)}, invocation,
            ir.Imm32(scratch_base));
        operation->ReplaceUsesWithAndRemove(replacement);
    }
}

void LowerWave64ScalarMaskPatterns(
    const std::vector<Wave64ControlledOperation>& patterns, u32 scratch_base,
    const ComputeRuntimeInfo& cs_info) {
    for (size_t index = 0; index < patterns.size(); ++index) {
        const Wave64ControlledOperation& pattern = patterns[index];
        const u32 pattern_scratch =
            scratch_base + static_cast<u32>(index) * Wave64ControlScratchBytes;
        InitializeWave64ControlState(pattern, pattern_scratch, cs_info);
        const auto insert_point =
            IR::Block::InstructionList::s_iterator_to(*pattern.operation);
        IR::IREmitter ir{*pattern.operation->GetParent(), insert_point};
        const IR::U32 invocation = LocalInvocationIndex(ir, cs_info);
        const IR::U32 replacement = ir.Wave64MaterializeScalarMaskWord(
            IR::U1{pattern.operation->Arg(0)}, IR::U1{pattern.operation->Arg(1)},
            IR::U32{pattern.operation->Arg(2)}, pattern.operation->Arg(3).U32(), invocation,
            ir.Imm32(pattern_scratch));
        pattern.operation->ReplaceUsesWithAndRemove(replacement);
    }
}

bool LowerGuestWaveGroupAnyOperations(const std::vector<IR::Inst*>& operations,
                                      ComputeRuntimeInfo& cs_info,
                                      const Profile& profile) {
    if (operations.empty()) {
        return true;
    }

    const u32 scratch_base = Common::AlignUp(cs_info.shared_memory_size, sizeof(u32));
    if (scratch_base > profile.max_shared_memory_size ||
        Wave64GroupAnyScratchBytes > profile.max_shared_memory_size - scratch_base) {
        LOG_INFO(Render_Recompiler,
                 "Wave64 subgroup lowering declined: LDS {} + GroupAny scratch {} exceeds host "
                 "max {}",
                 scratch_base, Wave64GroupAnyScratchBytes, profile.max_shared_memory_size);
        return false;
    }
    for (IR::Inst* operation : operations) {
        IR::IREmitter ir{*operation->GetParent(),
                         IR::Block::InstructionList::s_iterator_to(*operation)};
        const IR::U32 invocation = LocalInvocationIndex(ir, cs_info);
        const IR::U1 replacement = ir.Wave64GroupAny(
            IR::U1{operation->Arg(0)}, invocation, ir.Imm32(scratch_base));
        operation->ReplaceUsesWithAndRemove(replacement);
    }
    cs_info.shared_memory_size = scratch_base + Wave64GroupAnyScratchBytes;
    return true;
}

bool LowerGuestWaveGroupAny(IR::Program& program, ComputeRuntimeInfo& cs_info,
                            const Profile& profile) {
    std::vector<IR::Inst*> operations;
    for (IR::Block* block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::GuestWaveAny) {
                operations.push_back(&inst);
            }
        }
    }
    return LowerGuestWaveGroupAnyOperations(operations, cs_info, profile);
}

} // Anonymous namespace

void Wave64SubgroupLoweringPass(IR::Program& program, RuntimeInfo& runtime_info,
                                const Profile& profile, bool guest_exec_predication,
                                bool enable_split_wave64_scalar_control,
                                bool enable_wave64_read_first_lane) {
    if (program.info.stage != Stage::Compute || profile.subgroup_size == 0 ||
        profile.subgroup_size >= GcnWaveSize) {
        return;
    }

    auto& cs_info = runtime_info.cs_info;
    const u64 thread_count = static_cast<u64>(cs_info.workgroup_size[0]) *
                             cs_info.workgroup_size[1] * cs_info.workgroup_size[2];
    LowerSingleSubgroupPartialWaveOperations(program, cs_info, profile, thread_count);
    if (thread_count != GcnWaveSize) {
        return;
    }
    const auto exchange_blocks = WorkgroupExchangeBlocks(program);
    if (enable_split_wave64_scalar_control) {
        std::vector<IR::Inst*> guest_wave_votes;
        std::vector<IR::Inst*> exact_mask_captures;
        std::vector<IR::Inst*> exact_mask_materializations;
        for (IR::Block* block : program.blocks) {
            for (IR::Inst& inst : block->Instructions()) {
                switch (inst.GetOpcode()) {
                case IR::Opcode::GuestWaveAny:
                    guest_wave_votes.push_back(&inst);
                    break;
                case IR::Opcode::CaptureScalarMaskWord:
                    exact_mask_captures.push_back(&inst);
                    break;
                case IR::Opcode::MaterializeScalarMaskWord:
                    exact_mask_materializations.push_back(&inst);
                    break;
                default:
                    break;
                }
            }
        }

        const auto is_converged = [&exchange_blocks](const IR::Inst* inst) {
            return exchange_blocks.contains(inst->GetParent());
        };
        const size_t total_guest_wave_votes = guest_wave_votes.size();
        std::erase_if(guest_wave_votes, [&](const IR::Inst* vote) {
            return !is_converged(vote);
        });
        const size_t divergent_guest_wave_votes =
            total_guest_wave_votes - guest_wave_votes.size();
        const size_t total_exact_mask_captures = exact_mask_captures.size();
        if (!std::ranges::all_of(exact_mask_captures, is_converged)) {
            // Exact mask capture is one coherent representation change. If any producer is
            // divergent, leave every capture/materialization on the established native fallback
            // while still allowing independently safe scalar votes to be repaired.
            exact_mask_captures.clear();
            LOG_INFO(Render_Recompiler,
                     "Wave64 exact mask capture declined: producer is inside an "
                     "EXEC-divergent host region ({} captures)",
                     total_exact_mask_captures);
        }
        if (divergent_guest_wave_votes != 0) {
            LOG_INFO(Render_Recompiler,
                     "Wave64 scalar control leaves {} vote(s) native inside nonuniform "
                     "host control and lowers {} converged vote(s)",
                     divergent_guest_wave_votes, guest_wave_votes.size());
        }
        if (!guest_wave_votes.empty() || !exact_mask_captures.empty()) {
            const u32 scratch_base = Common::AlignUp(cs_info.shared_memory_size, sizeof(u32));
            if (scratch_base > profile.max_shared_memory_size ||
                Wave64GroupAnyScratchBytes > profile.max_shared_memory_size - scratch_base) {
                LOG_INFO(Render_Recompiler,
                         "Wave64 scalar control declined: LDS {} + scalar scratch {} exceeds "
                         "host max {}",
                         scratch_base, Wave64GroupAnyScratchBytes,
                         profile.max_shared_memory_size);
                return;
            }
            if (!LowerGuestWaveGroupAnyOperations(guest_wave_votes, cs_info, profile)) {
                return;
            }
            if (guest_wave_votes.empty()) {
                cs_info.shared_memory_size = scratch_base + Wave64GroupAnyScratchBytes;
            }
            for (IR::Inst* capture : exact_mask_captures) {
                const auto insert_point =
                    IR::Block::InstructionList::s_iterator_to(*capture);
                IR::IREmitter ir{*capture->GetParent(), insert_point};
                const IR::U32 invocation = LocalInvocationIndex(ir, cs_info);
                const IR::U32 replacement = ir.Wave64CaptureScalarMaskWord(
                    IR::U1{capture->Arg(0)}, capture->Arg(2).U32(), invocation,
                    ir.Imm32(scratch_base));
                capture->ReplaceUsesWithAndRemove(replacement);
            }
            if (!exact_mask_captures.empty()) {
                for (IR::Inst* materialize : exact_mask_materializations) {
                    materialize->ReplaceUsesWithAndRemove(materialize->Arg(2));
                }
            } else if (!exact_mask_materializations.empty()) {
                LOG_INFO(Render_Recompiler,
                         "Wave64 scalar control leaves {} uncaptured mask materializations "
                         "native while lowering {} votes",
                         exact_mask_materializations.size(), guest_wave_votes.size());
            }
            program.info.requires_subgroup_uniform_control_flow = true;
            LOG_INFO(Render_Recompiler,
                     "Wave64 scalar control applies: votes={} captures={} materializations={}",
                     guest_wave_votes.size(), exact_mask_captures.size(),
                     exact_mask_materializations.size());
        }
    }
    const bool has_structured_control = std::ranges::any_of(
        program.syntax_list, [](const IR::AbstractSyntaxNode& node) {
            using Type = IR::AbstractSyntaxNode::Type;
            return node.type == Type::If || node.type == Type::Loop ||
                   node.type == Type::Break || node.type == Type::Repeat;
        });
    size_t lockstep_cross_lane_count{};
    size_t lockstep_mask_count{};
    for (IR::Block* block : program.blocks) {
        for (const IR::Inst& inst : block->Instructions()) {
            switch (inst.GetOpcode()) {
            case IR::Opcode::ReadLane:
            case IR::Opcode::ReadFirstLane:
            case IR::Opcode::WriteLane:
            case IR::Opcode::CaptureScalarMaskWord:
                ++lockstep_cross_lane_count;
                break;
            case IR::Opcode::MaterializeScalarMaskWord:
                ++lockstep_cross_lane_count;
                ++lockstep_mask_count;
                break;
            default:
                break;
            }
        }
    }
    const bool has_cross_lane_state = lockstep_cross_lane_count != 0;
    // A whole-wave dispatcher is valid only when vector register and memory effects are
    // explicitly predicated by guest EXEC. The large-mask gate confines the more expensive
    // dispatcher to shaders whose split subgroup32 state cannot be represented natively.
    const bool lockstep_control = guest_exec_predication && has_structured_control &&
                                  has_cross_lane_state && lockstep_mask_count >= 128;
    if (lockstep_control) {
        LOG_INFO(Render_Recompiler,
                 "Wave64 lockstep control selected for shader {:#x} ({} blocks, {} cross-lane, "
                 "{} masks)",
                 program.info.pgm_hash, program.blocks.size(), lockstep_cross_lane_count,
                 lockstep_mask_count);
        program.info.emulate_wave64_control_flow = true;
        const u32 shared_before_vote = cs_info.shared_memory_size;
        if (!LowerGuestWaveGroupAny(program, cs_info, profile)) {
            program.info.emulate_wave64_control_flow = false;
            return;
        }
        if (cs_info.shared_memory_size == shared_before_vote) {
            const u32 scratch_base = Common::AlignUp(shared_before_vote, sizeof(u32));
            if (scratch_base > profile.max_shared_memory_size ||
                Wave64GroupAnyScratchBytes > profile.max_shared_memory_size - scratch_base) {
                program.info.emulate_wave64_control_flow = false;
                return;
            }
            cs_info.shared_memory_size = scratch_base + Wave64GroupAnyScratchBytes;
        }
    }
    std::vector<IR::Inst*> lane_ids;
    std::vector<IR::Inst*> read_lanes;
    std::vector<IR::Inst*> read_first_lanes;
    std::vector<IR::Inst*> converged_read_first_lanes;
    std::vector<IR::Inst*> write_lanes;
    std::vector<IR::Inst*> scalar_mask_captures;
    std::vector<IR::Inst*> scalar_mask_materializations;
    for (IR::Block* block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::LaneId) {
                lane_ids.push_back(&inst);
            } else if (inst.GetOpcode() == IR::Opcode::ReadLane) {
                // A workgroup barrier is valid only where every invocation is guaranteed to
                // execute it. Structured merge blocks are safe once the enclosing control flow
                // has reconverged; loop bodies, loop headers, and conditional bodies are not.
                if (!lockstep_control && !exchange_blocks.contains(block)) {
                    LOG_INFO(Render_Recompiler,
                             "Wave64 subgroup lowering declined: ReadLane in divergent block");
                    return;
                }
                read_lanes.push_back(&inst);
            } else if (inst.GetOpcode() == IR::Opcode::WriteLane) {
                write_lanes.push_back(&inst);
            } else if (inst.GetOpcode() == IR::Opcode::ReadFirstLane) {
                if (inst.HasUses()) {
                    if (lockstep_control || exchange_blocks.contains(block)) {
                        converged_read_first_lanes.push_back(&inst);
                    } else {
                        read_first_lanes.push_back(&inst);
                    }
                }
            } else if (inst.GetOpcode() == IR::Opcode::MaterializeScalarMaskWord) {
                scalar_mask_materializations.push_back(&inst);
            } else if (inst.GetOpcode() == IR::Opcode::CaptureScalarMaskWord) {
                scalar_mask_captures.push_back(&inst);
            } else if (IsUnsupportedWaveOperation(inst, profile)) {
                // Changing LaneId while leaving a wave operation native mixes workgroup-wide
                // and host-subgroup semantics. In particular, a wave32 host cannot service a
                // ReadLane from guest lane 32. Leave the shader internally consistent until a
                // complete wave64 exchange lowering is available.
                LOG_INFO(Render_Recompiler,
                         "Wave64 subgroup lowering declined: unsupported opcode {}",
                         static_cast<u32>(inst.GetOpcode()));
                return;
            }
        }
    }
    if (lockstep_control && !scalar_mask_materializations.empty()) {
        ASSERT(cs_info.shared_memory_size >= Wave64GroupAnyScratchBytes);
        const u32 scratch_base = cs_info.shared_memory_size - Wave64GroupAnyScratchBytes;
        for (IR::Inst* materialize : scalar_mask_materializations) {
            const auto insert_point =
                IR::Block::InstructionList::s_iterator_to(*materialize);
            IR::IREmitter ir{*materialize->GetParent(), insert_point};
            const IR::U32 invocation = LocalInvocationIndex(ir, cs_info);
            const IR::U32 replacement = ir.Wave64LockstepMaterializeScalarMaskWord(
                IR::U1{materialize->Arg(0)}, IR::U1{materialize->Arg(1)},
                IR::U32{materialize->Arg(2)}, materialize->Arg(3).U32(), invocation,
                ir.Imm32(scratch_base));
            materialize->ReplaceUsesWithAndRemove(replacement);
        }
        scalar_mask_materializations.clear();
    }
    const bool had_scalar_mask_materializations = !scalar_mask_materializations.empty();
    std::erase_if(scalar_mask_materializations, [](IR::Inst* materialize) {
        const IR::Value is_thread_bit = materialize->Arg(0).Resolve();
        if (!is_thread_bit.IsImmediate() || is_thread_bit.U1()) {
            return false;
        }
        materialize->ReplaceUsesWithAndRemove(materialize->Arg(2));
        return true;
    });
    if (had_scalar_mask_materializations) {
        // A subgroup32 ballot cannot provide both words of a guest wave64 scalar mask. Partly
        // synchronizing ReadFirstLane while leaving these words split has produced invalid
        // downstream resource state on real NVIDIA drivers. Fixed-lane spill/reload is
        // independent: WriteLane is a per-invocation select, while ReadLane sites admitted above
        // are workgroup-converged and can use a bounded exchange without changing scalar EXEC
        // control.
        for (IR::Inst* write_lane : write_lanes) {
            IR::IREmitter ir{*write_lane->GetParent(),
                             IR::Block::InstructionList::s_iterator_to(*write_lane)};
            const IR::U32 local_lane = LocalInvocationIndex(ir, cs_info);
            const IR::U32 target_lane =
                ir.BitwiseAnd(IR::U32{write_lane->Arg(2)}, ir.Imm32(GcnWaveSize - 1));
            const IR::U32 value{ir.Select(ir.IEqual(local_lane, target_lane),
                                         write_lane->Arg(1), write_lane->Arg(0))};
            write_lane->ReplaceUsesWithAndRemove(value);
        }

        if (!read_lanes.empty()) {
            const u32 scratch_base =
                Common::AlignUp(cs_info.shared_memory_size, sizeof(u32));
            if (scratch_base <= profile.max_shared_memory_size &&
                Wave64ExchangeScratchBytes <= profile.max_shared_memory_size - scratch_base) {
                for (IR::Inst* read_lane : read_lanes) {
                    IR::IREmitter ir{*read_lane->GetParent(),
                                     IR::Block::InstructionList::s_iterator_to(*read_lane)};
                    const IR::U32 local_lane = LocalInvocationIndex(ir, cs_info);
                    const IR::U32 local_offset = ir.IAdd(
                        ir.Imm32(scratch_base),
                        ir.ShiftLeftLogical(local_lane, ir.Imm32(2U)));
                    ir.WriteShared(32, IR::U32{read_lane->Arg(0)}, local_offset);
                    ir.Barrier();
                    const IR::U32 source_lane = ir.BitwiseAnd(
                        IR::U32{read_lane->Arg(1)}, ir.Imm32(GcnWaveSize - 1));
                    const IR::U32 source_offset = ir.IAdd(
                        ir.Imm32(scratch_base),
                        ir.ShiftLeftLogical(source_lane, ir.Imm32(2U)));
                    const IR::U32 value{ir.LoadShared(32, false, source_offset)};
                    ir.Barrier();
                    read_lane->ReplaceUsesWithAndRemove(value);
                }
                cs_info.shared_memory_size = scratch_base + Wave64ExchangeScratchBytes;
            }
        }

        // The shader remains split at scalar-mask materialization sites, but LaneId itself is a
        // guest-wave value. A wave32 host reports 0..31 in both subgroup halves; leaving that
        // native makes upper-half invocations select the low word of every saved EXEC/VCC mask.
        // Unsupported cross-half subgroup operations were rejected above, so using the linear
        // invocation index here restores the one-wave workgroup's PS4 lane numbering without
        // mixing it with an unlowered cross-half shuffle.
        if (!scalar_mask_materializations.empty()) {
            for (IR::Inst* lane_id : lane_ids) {
                IR::IREmitter ir{*lane_id->GetParent(),
                                 IR::Block::InstructionList::s_iterator_to(*lane_id)};
                lane_id->ReplaceUsesWithAndRemove(LocalInvocationIndex(ir, cs_info));
            }
        }

        // Scalar mask words may still be split while scalar branch votes are synchronized. A
        // converged ReadFirstLane is independent of those saved words: its explicit EXEC value
        // identifies active guest lanes, and a small workgroup exchange combines the two host
        // halves. Do not partly lower shaders which also contain a divergent ReadFirstLane; that
        // still needs the separate retirement protocol to model guest EXEC precisely.
        if ((enable_wave64_read_first_lane || enable_split_wave64_scalar_control) &&
            read_first_lanes.empty() &&
            !converged_read_first_lanes.empty()) {
            const u32 scratch_base =
                Common::AlignUp(cs_info.shared_memory_size, sizeof(u32));
            if (scratch_base <= profile.max_shared_memory_size &&
                Wave64ReadFirstActiveScratchBytes <=
                    profile.max_shared_memory_size - scratch_base) {
                LowerConvergedReadFirstActiveLanes(converged_read_first_lanes, scratch_base,
                                                   cs_info);
                cs_info.shared_memory_size =
                    scratch_base + Wave64ReadFirstActiveScratchBytes;
                converged_read_first_lanes.clear();
            } else {
                LOG_INFO(Render_Recompiler,
                         "Wave64 scalar control leaves converged ReadFirstLane native: LDS {} + "
                         "scratch {} exceeds host max {}",
                         scratch_base, Wave64ReadFirstActiveScratchBytes,
                         profile.max_shared_memory_size);
            }
        }

        program.info.requires_subgroup_uniform_control_flow = true;
        if (scalar_mask_materializations.size() >= 32) {
            const size_t converged_captures = std::ranges::count_if(
                scalar_mask_captures, [&exchange_blocks](const IR::Inst* capture) {
                    return exchange_blocks.contains(capture->GetParent());
                });
            LOG_INFO(Render_Recompiler,
                     "Wave64 scalar mask producers: converged={} total={}", converged_captures,
                     scalar_mask_captures.size());
            std::map<IR::Opcode, u32> consumer_counts;
            for (const IR::Inst* materialize : scalar_mask_materializations) {
                for (const IR::Use& use : materialize->Uses()) {
                    ++consumer_counts[use.user->GetOpcode()];
                }
            }
            std::string consumers;
            for (const auto& [opcode, count] : consumer_counts) {
                consumers += fmt::format("{}={} ", opcode, count);
            }
            LOG_INFO(Render_Recompiler, "Wave64 scalar mask consumers: {}", consumers);
        }
        LOG_INFO(Render_Recompiler,
                 "Wave64 subgroup lowering declined: {} scalar mask state ({} materializations "
                 "remain)",
                 scalar_mask_materializations.empty() ? "scalar-only" : "split",
                 scalar_mask_materializations.size());
        return;
    }
    const auto read_first_patterns = read_first_lanes.empty()
                                         ? std::optional{
                                               std::vector<Wave64ControlledOperation>{}}
                                         : FindWave64ControlledOperations(
                                               program, IR::Opcode::ReadFirstLane, false);
    if (!read_first_patterns || read_first_patterns->size() != read_first_lanes.size()) {
        LOG_INFO(Render_Recompiler,
                 "Wave64 subgroup lowering declined: unsupported ReadFirstLane control flow");
        return;
    }
    if (std::ranges::any_of(read_first_lanes, [](const IR::Inst* operation) {
            const IR::Value active = operation->Arg(1).Resolve();
            return !active.IsImmediate() || !active.U1();
        })) {
        // The controlled retirement protocol currently tracks physical subgroup liveness. It is
        // safe for synthetic always-active IR tests, but a divergent dynamic guest EXEC needs an
        // additional per-generation availability exchange. Preserve native behavior until that
        // protocol is complete.
        LOG_INFO(Render_Recompiler,
                 "Wave64 subgroup lowering declined: divergent ReadFirstLane has dynamic EXEC");
        return;
    }
    if (!read_first_patterns->empty() &&
        (!read_lanes.empty() || !converged_read_first_lanes.empty())) {
        // Controlled exchanges preserve guest EXEC participation with per-half generations.
        // Barrier-based exchanges require every physical invocation to remain live. Combining
        // the protocols after structured divergence has caused a host half to wait forever on
        // real drivers, so fail closed until physical liveness across the merge is proven.
        LOG_INFO(Render_Recompiler,
                 "Wave64 subgroup lowering declined: mixed controlled and barrier exchanges");
        return;
    }
    const auto scalar_mask_patterns =
        std::optional{std::vector<Wave64ControlledOperation>{}};
    if (lane_ids.empty() && read_lanes.empty() && read_first_lanes.empty() &&
        converged_read_first_lanes.empty() && write_lanes.empty()) {
        return;
    }

    const u32 control_scratch_base =
        Common::AlignUp(cs_info.shared_memory_size, sizeof(u32));
    const u64 control_scratch_bytes =
        static_cast<u64>(read_first_patterns->size() + scalar_mask_patterns->size()) *
        Wave64ControlScratchBytes;
    if (control_scratch_base > profile.max_shared_memory_size ||
        control_scratch_bytes > profile.max_shared_memory_size - control_scratch_base) {
        LOG_INFO(Render_Recompiler,
                 "Wave64 subgroup lowering declined: LDS {} + control scratch {} exceeds host "
                 "max {}",
                 control_scratch_base, control_scratch_bytes, profile.max_shared_memory_size);
        return;
    }
    const u32 control_scratch_end =
        control_scratch_base + static_cast<u32>(control_scratch_bytes);
    const u32 scratch_base = Common::AlignUp(control_scratch_end, sizeof(u32));
    const u32 exchange_scratch_bytes = !read_lanes.empty()
                                               ? Wave64ExchangeScratchBytes
                                               : (!converged_read_first_lanes.empty()
                                                      ? Wave64ReadFirstActiveScratchBytes
                                                      : 0U);
    if (exchange_scratch_bytes != 0U &&
        (scratch_base > profile.max_shared_memory_size ||
         exchange_scratch_bytes > profile.max_shared_memory_size - scratch_base)) {
        LOG_INFO(Render_Recompiler,
                 "Wave64 subgroup lowering declined: LDS {} + scratch {} exceeds host max {}",
                 scratch_base, exchange_scratch_bytes, profile.max_shared_memory_size);
        return;
    }

    LOG_INFO(Render_Recompiler,
             "Wave64 subgroup lowering applies: lanes={} read_lanes={} read_first_lanes={} "
             "converged_read_first_lanes={} write_lanes={} scalar_masks={} guest_lds={} "
             "host_max={}",
             lane_ids.size(), read_lanes.size(), read_first_lanes.size(),
             converged_read_first_lanes.size(), write_lanes.size(),
             scalar_mask_materializations.size(), cs_info.shared_memory_size,
             profile.max_shared_memory_size);

    if (!read_first_patterns->empty()) {
        LowerWave64ReadFirstLanePatterns(*read_first_patterns, control_scratch_base, cs_info);
    }
    if (!scalar_mask_patterns->empty()) {
        const u32 scalar_mask_scratch_base =
            control_scratch_base + static_cast<u32>(read_first_patterns->size()) *
                                       Wave64ControlScratchBytes;
        LowerWave64ScalarMaskPatterns(*scalar_mask_patterns, scalar_mask_scratch_base, cs_info);
    }
    if (!read_first_patterns->empty() || !scalar_mask_patterns->empty()) {
        cs_info.shared_memory_size = control_scratch_end;
    }
    if (!scalar_mask_materializations.empty()) {
        // Scalar branch liveness and ReadFirstLane are now synchronized, but materializing a full
        // guest mask still needs both 32-bit words rather than the current subgroup ballot. Keep
        // that independent transformation fail-closed while retaining the safe control repairs.
        program.info.requires_subgroup_uniform_control_flow = true;
        for (IR::Inst* materialize : scalar_mask_materializations) {
            const IR::Value is_thread_bit = materialize->Arg(0).Resolve();
            if (is_thread_bit.IsImmediate() && !is_thread_bit.U1()) {
                materialize->ReplaceUsesWithAndRemove(materialize->Arg(2));
            }
        }
        LOG_INFO(Render_Recompiler,
                 "Wave64 subgroup lowering stops after control synchronization: scalar mask "
                 "materialization remains native");
        return;
    }
    if (!scalar_mask_materializations.empty()) {
        // The owning host half materializes its corresponding guest scalar-mask word. The
        // structured control protocol above keeps both halves on the same guest-wave path while
        // ReadFirstLane exchanges the selected value without a divergent workgroup barrier.
        program.info.requires_subgroup_uniform_control_flow = true;
    }

    for (IR::Inst* write_lane : write_lanes) {
        IR::IREmitter ir{*write_lane->GetParent(),
                         IR::Block::InstructionList::s_iterator_to(*write_lane)};
        const IR::U32 local_lane = LocalInvocationIndex(ir, cs_info);
        const IR::U32 target_lane =
            ir.BitwiseAnd(IR::U32{write_lane->Arg(2)}, ir.Imm32(GcnWaveSize - 1));
        const IR::U32 value{ir.Select(ir.IEqual(local_lane, target_lane),
                                     write_lane->Arg(1), write_lane->Arg(0))};
        write_lane->ReplaceUsesWithAndRemove(value);
    }

    for (IR::Inst* read_lane : read_lanes) {
        IR::IREmitter ir{*read_lane->GetParent(),
                         IR::Block::InstructionList::s_iterator_to(*read_lane)};
        const IR::U32 local_lane = LocalInvocationIndex(ir, cs_info);
        const IR::U32 local_offset = ir.IAdd(
            ir.Imm32(scratch_base), ir.ShiftLeftLogical(local_lane, ir.Imm32(2U)));
        ir.WriteShared(32, IR::U32{read_lane->Arg(0)}, local_offset);
        ir.Barrier();
        const IR::U32 source_lane =
            ir.BitwiseAnd(IR::U32{read_lane->Arg(1)}, ir.Imm32(GcnWaveSize - 1));
        const IR::U32 source_offset = ir.IAdd(
            ir.Imm32(scratch_base), ir.ShiftLeftLogical(source_lane, ir.Imm32(2U)));
        const IR::U32 value{ir.LoadShared(32, false, source_offset)};
        ir.Barrier();
        read_lane->ReplaceUsesWithAndRemove(value);
    }

    LowerConvergedReadFirstActiveLanes(converged_read_first_lanes, scratch_base, cs_info);

    if (exchange_scratch_bytes != 0U) {
        cs_info.shared_memory_size = scratch_base + exchange_scratch_bytes;
    }

    for (IR::Inst* lane_id : lane_ids) {
        IR::IREmitter ir{*lane_id->GetParent(),
                         IR::Block::InstructionList::s_iterator_to(*lane_id)};
        lane_id->ReplaceUsesWithAndRemove(LocalInvocationIndex(ir, cs_info));
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

void Wave64ProtocolCleanupPass(IR::Program& program) {
    std::unordered_set<u32> live_scratch_bases;
    for (IR::Block* block : program.blocks) {
        for (const IR::Inst& inst : block->Instructions()) {
            const IR::Opcode opcode = inst.GetOpcode();
            if (opcode != IR::Opcode::Wave64ReadFirstLane &&
                opcode != IR::Opcode::Wave64MaterializeScalarMaskWord) {
                continue;
            }
            const size_t scratch_arg =
                opcode == IR::Opcode::Wave64ReadFirstLane ? 3U : 5U;
            const IR::Value scratch = inst.Arg(scratch_arg).Resolve();
            if (scratch.IsImmediate()) {
                live_scratch_bases.insert(scratch.U32());
            }
        }
    }

    for (IR::Block* block : program.blocks) {
        auto it = block->begin();
        while (it != block->end()) {
            const IR::Opcode opcode = it->GetOpcode();
            if (opcode != IR::Opcode::Wave64LaneRetirementInit &&
                opcode != IR::Opcode::Wave64LaneRetirementSync) {
                ++it;
                continue;
            }
            const IR::Value scratch = it->Arg(2).Resolve();
            if (!scratch.IsImmediate() || live_scratch_bases.contains(scratch.U32())) {
                ++it;
                continue;
            }
            it->Invalidate();
            it = block->Instructions().erase(it);
        }
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
