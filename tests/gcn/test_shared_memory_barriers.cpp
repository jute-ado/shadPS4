// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <optional>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "spirv/unified1/spirv.hpp"

namespace {

size_t CountOpcode(const Shader::IR::Program& program, Shader::IR::Opcode opcode) {
    size_t count{};
    for (const Shader::IR::Block* block : program.blocks) {
        count += std::ranges::count_if(block->Instructions(), [opcode](const auto& inst) {
            return inst.GetOpcode() == opcode;
        });
    }
    return count;
}

std::optional<u32> BallotExtractIndex(const std::vector<u32>& spirv) {
    u32 ballot_id{};
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 word_count = word >> 16;
        const auto opcode = static_cast<spv::Op>(word & 0xffff);
        if (opcode == spv::Op::OpGroupNonUniformBallot) {
            ballot_id = spirv[offset + 2];
        } else if (opcode == spv::Op::OpCompositeExtract && word_count == 5 &&
                   ballot_id != 0 && spirv[offset + 3] == ballot_id) {
            return spirv[offset + 4];
        }
        if (word_count == 0) {
            return std::nullopt;
        }
        offset += word_count;
    }
    return std::nullopt;
}

size_t CountSpirvOpcode(const std::vector<u32>& spirv, spv::Op expected) {
    size_t count{};
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 word_count = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == expected) {
            ++count;
        }
        if (word_count == 0) {
            return count;
        }
        offset += word_count;
    }
    return count;
}

bool LoopMergesImmediatelyPrecedeBranch(const std::vector<u32>& spirv) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word_count = spirv[offset] >> 16;
        if (word_count == 0) {
            return false;
        }
        const auto opcode = static_cast<spv::Op>(spirv[offset] & 0xffff);
        const size_t next = offset + word_count;
        if (opcode == spv::Op::OpLoopMerge) {
            if (next >= spirv.size()) {
                return false;
            }
            const auto next_opcode = static_cast<spv::Op>(spirv[next] & 0xffff);
            if (next_opcode != spv::Op::OpBranch &&
                next_opcode != spv::Op::OpBranchConditional) {
                return false;
            }
        }
        offset = next;
    }
    return true;
}

bool SelectionMergesDoNotTargetLoopContinueBlocks(const std::vector<u32>& spirv) {
    std::vector<u32> loop_continue_blocks;
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word_count = spirv[offset] >> 16;
        if (word_count == 0) {
            return false;
        }
        if (static_cast<spv::Op>(spirv[offset] & 0xffff) == spv::Op::OpLoopMerge) {
            loop_continue_blocks.push_back(spirv[offset + 2]);
        }
        offset += word_count;
    }
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word_count = spirv[offset] >> 16;
        if (word_count == 0) {
            return false;
        }
        if (static_cast<spv::Op>(spirv[offset] & 0xffff) == spv::Op::OpSelectionMerge &&
            std::ranges::find(loop_continue_blocks, spirv[offset + 1]) !=
                loop_continue_blocks.end()) {
            return false;
        }
        offset += word_count;
    }
    return true;
}

bool HasExecutionMode(const std::vector<u32>& spirv, spv::ExecutionMode expected) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 word_count = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpExecutionMode &&
            word_count >= 3 && spirv[offset + 2] == static_cast<u32>(expected)) {
            return true;
        }
        if (word_count == 0) {
            return false;
        }
        offset += word_count;
    }
    return false;
}

std::optional<std::pair<spv::LoopControlMask, u32>> FirstLoopControl(
    const std::vector<u32>& spirv) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 word_count = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpLoopMerge && word_count >= 4) {
            const auto control = static_cast<spv::LoopControlMask>(spirv[offset + 3]);
            const u32 partial_count = word_count >= 5 ? spirv[offset + 4] : 0;
            return std::pair{control, partial_count};
        }
        if (word_count == 0) {
            return std::nullopt;
        }
        offset += word_count;
    }
    return std::nullopt;
}

bool HasBuiltIn(const std::vector<u32>& spirv, spv::BuiltIn expected) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 word_count = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpDecorate && word_count >= 4 &&
            spirv[offset + 2] == static_cast<u32>(spv::Decoration::DecorationBuiltIn) &&
            spirv[offset + 3] == static_cast<u32>(expected)) {
            return true;
        }
        if (word_count == 0) {
            return false;
        }
        offset += word_count;
    }
    return false;
}

bool HasCapability(const std::vector<u32>& spirv, spv::Capability expected) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 word_count = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpCapability && word_count == 2 &&
            spirv[offset + 1] == static_cast<u32>(expected)) {
            return true;
        }
        if (word_count == 0) {
            return false;
        }
        offset += word_count;
    }
    return false;
}

bool BuiltInHasDecoration(const std::vector<u32>& spirv, spv::BuiltIn built_in,
                          spv::Decoration decoration) {
    std::optional<u32> variable;
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 word_count = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpDecorate && word_count >= 4 &&
            spirv[offset + 2] == static_cast<u32>(spv::Decoration::DecorationBuiltIn) &&
            spirv[offset + 3] == static_cast<u32>(built_in)) {
            variable = spirv[offset + 1];
        }
        if (word_count == 0) {
            return false;
        }
        offset += word_count;
    }
    if (!variable) {
        return false;
    }
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 word_count = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpDecorate && word_count >= 3 &&
            spirv[offset + 1] == *variable &&
            spirv[offset + 2] == static_cast<u32>(decoration)) {
            return true;
        }
        if (word_count == 0) {
            return false;
        }
        offset += word_count;
    }
    return false;
}

TEST(SubgroupBuiltIns, VertexLaneIdIsNotFlatDecorated) {
    Shader::Info info{};
    info.stage = Shader::Stage::Vertex;
    info.l_stage = Shader::LogicalStage::Vertex;
    Shader::IR::Program program{info};
    Shader::Pools pools{};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());
    Shader::IR::IREmitter ir{*block};
    ir.Prologue();
    ir.Reference(ir.LaneId());
    ir.Epilogue();

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Vertex);
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    Shader::Optimization::CollectShaderInfoPass(program, profile);
    Shader::Backend::Bindings bindings{};
    const auto spirv =
        Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);

    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltIn::BuiltInSubgroupLocalInvocationId));
    EXPECT_FALSE(BuiltInHasDecoration(spirv, spv::BuiltIn::BuiltInSubgroupLocalInvocationId,
                                      spv::Decoration::DecorationFlat));
}

TEST(SubgroupBuiltIns, FragmentLaneIdIsFlatDecorated) {
    Shader::Info info{};
    info.stage = Shader::Stage::Fragment;
    info.l_stage = Shader::LogicalStage::Fragment;
    Shader::IR::Program program{info};
    Shader::Pools pools{};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());
    Shader::IR::IREmitter ir{*block};
    ir.Prologue();
    ir.Reference(ir.LaneId());
    ir.Epilogue();

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Fragment);
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    Shader::Optimization::CollectShaderInfoPass(program, profile);
    Shader::Backend::Bindings bindings{};
    const auto spirv =
        Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);

    EXPECT_TRUE(BuiltInHasDecoration(spirv, spv::BuiltIn::BuiltInSubgroupLocalInvocationId,
                                     spv::Decoration::DecorationFlat));
}

TEST(Wave64SubgroupLoweringPass, UsesOneWorkgroupWaveForLaneIdOnlyShader) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U32 lane = ir.LaneId();
    ir.Reference(lane);
    ir.Epilogue();

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {4, 2, 8};
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile, false, true);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::LaneId), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::GetAttributeU32), 3U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Barrier), 0U);
    EXPECT_EQ(runtime_info.cs_info.shared_memory_size, 0U);
}

TEST(Wave64SubgroupLoweringPass, LowersConvergedGuestWaveAnyWithoutDispatcher) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;
    Shader::IR::IREmitter ir{*block};
    ir.Prologue();
    ir.Reference(ir.GuestWaveAny(ir.Imm1(true)));
    ir.Epilogue();

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile, false, true);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::GuestWaveAny), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64GroupAny), 1U);
    EXPECT_FALSE(program.info.emulate_wave64_control_flow);
    EXPECT_EQ(runtime_info.cs_info.shared_memory_size, 2U * sizeof(u32));

    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());
    Shader::Optimization::CollectShaderInfoPass(program, profile);
    Shader::Backend::Bindings bindings{};
    const auto spirv =
        Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
    // The fixed-size guest-wave vote rendezvous is independent of the rejected block dispatcher.
    EXPECT_TRUE(HasCapability(spirv, spv::Capability::CapabilityGroupNonUniformVote));
    // Publishing once per host subgroup uses OpGroupNonUniformBroadcastFirst to select its
    // first active invocation. SPIR-V classifies that operation under GroupNonUniformBallot.
    EXPECT_TRUE(HasCapability(spirv, spv::Capability::CapabilityGroupNonUniformBallot));
    EXPECT_EQ(CountSpirvOpcode(spirv, spv::Op::OpControlBarrier), 2U);
}

TEST(Wave64SubgroupLoweringPass, LowersVoteWhileLeavingUncapturedMaskNative) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;
    Shader::IR::IREmitter ir{*block};
    ir.Prologue();
    ir.Reference(ir.GuestWaveAny(ir.Imm1(true)));
    ir.Reference(ir.MaterializeScalarMaskWord(ir.Imm1(true), ir.Imm1(true), ir.Imm32(7), 0U));
    ir.Epilogue();

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile, false, true);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::GuestWaveAny), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64GroupAny), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::MaterializeScalarMaskWord), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64CaptureScalarMaskWord), 0U);
    EXPECT_EQ(runtime_info.cs_info.shared_memory_size, 2U * sizeof(u32));
}

TEST(Wave64SubgroupLoweringPass,
     LowersExecAwareReadFirstLaneWhileLeavingUncapturedMaskNative) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;
    Shader::IR::IREmitter ir{*block};
    ir.Prologue();
    const Shader::IR::U32 lane = ir.LaneId();
    const Shader::IR::U1 active = ir.IGreaterThanEqual(lane, ir.Imm32(4U), false);
    ir.Reference(ir.GuestWaveAny(active));
    ir.Reference(ir.ReadFirstLane(lane, active));
    ir.Reference(ir.MaterializeScalarMaskWord(ir.Imm1(true), active, ir.Imm32(7), 0U));
    ir.Epilogue();

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile, false, true);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::GuestWaveAny), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64GroupAny), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ReadFirstLane), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64ReadFirstActiveLane), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::MaterializeScalarMaskWord), 1U);
    EXPECT_EQ(runtime_info.cs_info.shared_memory_size, 6U * sizeof(u32));

    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());
    Shader::Optimization::CollectShaderInfoPass(program, profile);
    Shader::Backend::Bindings bindings{};
    const auto spirv =
        Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
    EXPECT_TRUE(HasCapability(spirv, spv::Capability::CapabilityGroupNonUniformBallot));
    EXPECT_EQ(CountSpirvOpcode(spirv, spv::Op::OpControlBarrier), 4U);
}

TEST(Wave64SubgroupLoweringPass, LowersExecAwareReadFirstLaneWithoutChangingScalarVotes) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;
    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U32 lane = ir.LaneId();
    const Shader::IR::U1 active = ir.IGreaterThanEqual(lane, ir.Imm32(37U), false);
    ir.Reference(ir.GuestWaveAny(active));
    ir.Reference(ir.ReadFirstLane(lane, active));
    ir.Reference(ir.MaterializeScalarMaskWord(ir.Imm1(true), active, ir.Imm32(7), 1U));
    ir.Epilogue();

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile, false,
                                                      false, true);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::GuestWaveAny), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64GroupAny), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ReadFirstLane), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64ReadFirstActiveLane), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::MaterializeScalarMaskWord), 1U);
    EXPECT_EQ(runtime_info.cs_info.shared_memory_size, 4U * sizeof(u32));
}

TEST(Wave64SubgroupLoweringPass, LeavesVoteAfterNonuniformBreakNativeButLowersMergeVote) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};
    const auto make_block = [&] {
        auto* const block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        return block;
    };
    auto* const header = make_block();
    auto* const body = make_block();
    auto* const after_break = make_block();
    auto* const continue_block = make_block();
    auto* const merge = make_block();

    Shader::IR::IREmitter body_ir{*body};
    const Shader::IR::U1 lane_break = body_ir.ConditionRef(
        body_ir.ILessThan(body_ir.LaneId(), body_ir.Imm32(32U), false));
    Shader::IR::IREmitter after_ir{*after_break};
    after_ir.Reference(after_ir.GuestWaveAny(after_ir.Imm1(true)));
    Shader::IR::IREmitter continue_ir{*continue_block};
    const Shader::IR::U1 repeat_cond = continue_ir.ConditionRef(continue_ir.Imm1(false));
    Shader::IR::IREmitter merge_ir{*merge};
    merge_ir.Reference(merge_ir.GuestWaveAny(merge_ir.Imm1(true)));
    merge_ir.Epilogue();

    const auto add_block = [&](Shader::IR::Block* block) {
        auto& node = program.syntax_list.emplace_back();
        node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
    };
    add_block(header);
    auto& loop = program.syntax_list.emplace_back();
    loop.type = Shader::IR::AbstractSyntaxNode::Type::Loop;
    loop.data.loop = {.body = body,
                      .continue_block = continue_block,
                      .merge = merge,
                      .is_guest_wave_uniform = true};
    add_block(body);
    auto& break_node = program.syntax_list.emplace_back();
    break_node.type = Shader::IR::AbstractSyntaxNode::Type::Break;
    break_node.data.break_node = {.cond = lane_break,
                                  .merge = merge,
                                  .skip = after_break,
                                  .is_guest_wave_uniform = false};
    add_block(after_break);
    add_block(continue_block);
    auto& repeat = program.syntax_list.emplace_back();
    repeat.type = Shader::IR::AbstractSyntaxNode::Type::Repeat;
    repeat.data.repeat = {.cond = repeat_cond,
                          .loop_header = header,
                          .merge = merge,
                          .is_guest_wave_uniform = true};
    add_block(merge);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile, false, true);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::GuestWaveAny), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64GroupAny), 1U);
    EXPECT_EQ(runtime_info.cs_info.shared_memory_size, 2U * sizeof(u32));
}

TEST(Wave64SubgroupLoweringPass, CapturesExactMaskWordAtConvergedProducer) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;
    Shader::IR::IREmitter ir{*block};
    ir.Prologue();
    const Shader::IR::U1 bit = ir.INotEqual(ir.LaneId(), ir.Imm32(63U));
    const Shader::IR::U32 captured =
        ir.CaptureScalarMaskWord(bit, ir.Imm32(0xdeadbeef), 1U);
    ir.Reference(ir.MaterializeScalarMaskWord(ir.Imm1(true), bit, captured, 1U));
    ir.Epilogue();

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile, false, true);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::CaptureScalarMaskWord), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64CaptureScalarMaskWord), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::MaterializeScalarMaskWord), 0U);
    EXPECT_FALSE(program.info.emulate_wave64_control_flow);
    EXPECT_EQ(runtime_info.cs_info.shared_memory_size, 2U * sizeof(u32));

    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());
    Shader::Optimization::CollectShaderInfoPass(program, profile);
    Shader::Backend::Bindings bindings{};
    const auto spirv =
        Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
    // A closing barrier prevents a faster invocation from overwriting reused scratch while a
    // slower invocation still consumes the captured word.
    EXPECT_EQ(CountSpirvOpcode(spirv, spv::Op::OpControlBarrier), 2U);
}

TEST(Wave64SubgroupLoweringPass, DeclinesMaskCaptureInsideExecDivergentRegion) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const entry = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {entry, body, merge};
    Shader::IR::IREmitter entry_ir{*entry};
    const Shader::IR::U1 condition = entry_ir.ConditionRef(
        entry_ir.ILessThan(entry_ir.LaneId(), entry_ir.Imm32(32U), false));
    Shader::IR::IREmitter body_ir{*body};
    const Shader::IR::U32 captured =
        body_ir.CaptureScalarMaskWord(body_ir.Imm1(true), body_ir.Imm32(0), 0U);
    body_ir.Reference(
        body_ir.MaterializeScalarMaskWord(body_ir.Imm1(true), body_ir.Imm1(true), captured, 0U));
    Shader::IR::IREmitter{*merge}.Epilogue();

    const auto add_block = [&](Shader::IR::Block* block) {
        auto& node = program.syntax_list.emplace_back();
        node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
    };
    add_block(entry);
    auto& if_node = program.syntax_list.emplace_back();
    if_node.type = Shader::IR::AbstractSyntaxNode::Type::If;
    if_node.data.if_node = {.cond = condition,
                            .body = body,
                            .merge = merge,
                            .is_guest_wave_uniform = false};
    add_block(body);
    auto& end_if = program.syntax_list.emplace_back();
    end_if.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
    end_if.data.end_if = {.merge = merge, .is_guest_wave_uniform = false};
    add_block(merge);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile, false, true);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::CaptureScalarMaskWord), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64CaptureScalarMaskWord), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::MaterializeScalarMaskWord), 1U);
    EXPECT_EQ(runtime_info.cs_info.shared_memory_size, 0U);
}

TEST(Wave64SubgroupLoweringPass, DeclinesWholeWaveDispatcherWithoutExecPredication) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const entry = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {entry, body, merge};
    Shader::IR::IREmitter entry_ir{*entry};
    entry_ir.Prologue();
    const Shader::IR::U1 condition =
        entry_ir.ConditionRef(entry_ir.GuestWaveAny(entry_ir.Imm1(true)));
    Shader::IR::IREmitter body_ir{*body};
    constexpr size_t MaskWalkSize = 128;
    for (size_t index = 0; index < MaskWalkSize; ++index) {
        body_ir.Reference(body_ir.MaterializeScalarMaskWord(
            body_ir.Imm1(true), body_ir.Imm1(true), body_ir.Imm32(0), index & 1));
    }
    Shader::IR::IREmitter merge_ir{*merge};
    merge_ir.Epilogue();

    auto add_block = [&](Shader::IR::Block* block) {
        auto& node = program.syntax_list.emplace_back();
        node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
    };
    add_block(entry);
    auto& if_node = program.syntax_list.emplace_back();
    if_node.type = Shader::IR::AbstractSyntaxNode::Type::If;
    if_node.data.if_node = {.cond = condition, .body = body, .merge = merge};
    add_block(body);
    auto& end_if = program.syntax_list.emplace_back();
    end_if.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
    end_if.data.end_if.merge = merge;
    add_block(merge);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_FALSE(program.info.emulate_wave64_control_flow);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::GuestWaveAny), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64GroupAny), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::MaterializeScalarMaskWord), MaskWalkSize);
    EXPECT_EQ(CountOpcode(program,
                          Shader::IR::Opcode::Wave64LockstepMaterializeScalarMaskWord),
              0U);

    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());
    Shader::Optimization::CollectShaderInfoPass(program, profile);
    Shader::Backend::Bindings bindings{};
    const auto spirv =
        Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
    EXPECT_EQ(CountSpirvOpcode(spirv, spv::Op::OpSwitch), 0U);

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile, true);

    EXPECT_TRUE(program.info.emulate_wave64_control_flow);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64GroupAny), 1U);
    EXPECT_EQ(CountOpcode(program,
                          Shader::IR::Opcode::Wave64LockstepMaterializeScalarMaskWord),
              MaskWalkSize);

    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());
    Shader::Optimization::SsaDestroyPass(program, true);
    Shader::Optimization::CollectShaderInfoPass(program, profile);
    const auto lockstep_spirv =
        Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
    EXPECT_TRUE(LoopMergesImmediatelyPrecedeBranch(lockstep_spirv));
    EXPECT_TRUE(SelectionMergesDoNotTargetLoopContinueBlocks(lockstep_spirv));
    // Workgroup barriers inside guest blocks require every physical invocation to scan the same
    // blocks. A per-lane switch can be structurally valid SPIR-V while dynamically violating
    // barrier convergence when the two host subgroup32 halves choose different cases.
    EXPECT_EQ(CountSpirvOpcode(lockstep_spirv, spv::Op::OpSwitch), 0U);
    EXPECT_GT(CountSpirvOpcode(lockstep_spirv, spv::Op::OpAtomicOr), 0U);
}

TEST(Wave64SubgroupLoweringPass, DeclinedGuestWaveAnyPreservesSplitHostControl) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());
    Shader::IR::IREmitter ir{*block};
    ir.Prologue();
    ir.Reference(ir.GuestWaveAny(ir.Imm1(true)));
    ir.Epilogue();

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {4, 4, 1};
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    Shader::Optimization::CollectShaderInfoPass(program, profile);
    Shader::Backend::Bindings bindings{};
    const auto spirv =
        Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
    EXPECT_EQ(CountSpirvOpcode(spirv, spv::Op::OpGroupNonUniformAny), 0U);
}

TEST(Wave64SubgroupLoweringPass, SplitSubgroupMaskUsesGuestWaveLaneId) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());

    Shader::IR::IREmitter ir{*block};
    ir.Prologue();
    ir.Reference(ir.LaneId());
    ir.Reference(ir.MaterializeScalarMaskWord(ir.Imm1(true), ir.Imm1(true), ir.Imm32(0), 1));
    ir.Reference(ir.MaterializeScalarMaskWord(ir.Imm1(true), ir.Imm1(false), ir.Imm32(0), 0));
    ir.Epilogue();

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::LaneId), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::GetAttributeU32), 3U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::MaterializeScalarMaskWord), 2U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Ballot), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::WriteSharedU32), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::LoadSharedU32), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Barrier), 0U);
    EXPECT_EQ(runtime_info.cs_info.shared_memory_size, 0U);
    Shader::Optimization::CollectShaderInfoPass(program, profile);

    Shader::Backend::Bindings bindings{};
    const auto spirv =
        Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
    EXPECT_EQ(BallotExtractIndex(spirv), 1U);
    EXPECT_TRUE(HasCapability(spirv, spv::Capability::CapabilityGroupNonUniformBallot));
    EXPECT_EQ(CountSpirvOpcode(spirv, spv::Op::OpGroupNonUniformBallot), 2U);
    EXPECT_EQ(CountSpirvOpcode(spirv, spv::Op::OpGroupNonUniformBroadcast), 0U);
}

TEST(Wave64SubgroupLoweringPass, LowersConvergedLaneSpillDespiteSplitScalarMask) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;

    Shader::IR::IREmitter ir{*block};
    ir.Prologue();
    const Shader::IR::U32 lane = ir.LaneId();
    const Shader::IR::U32 written = ir.WriteLane(lane, ir.Imm32(0x12345678), ir.Imm32(32));
    ir.Reference(ir.ReadLane(written, ir.Imm32(32)));
    ir.Reference(ir.MaterializeScalarMaskWord(ir.Imm1(true), ir.Imm1(true), ir.Imm32(0), 1));
    ir.Epilogue();

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::WriteLane), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ReadLane), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::SelectU32), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::WriteSharedU32), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::LoadSharedU32), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Barrier), 2U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::MaterializeScalarMaskWord), 1U);
}

TEST(Wave64SubgroupLoweringPass, ScalarOnlyMaskKeepsUniformControlWithoutBallot) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());

    Shader::IR::IREmitter ir{*block};
    ir.Prologue();
    ir.Reference(ir.LaneId());
    ir.Reference(ir.MaterializeScalarMaskWord(ir.Imm1(false), ir.Imm1(true), ir.Imm32(9), 0));
    ir.Epilogue();

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;
    profile.supports_subgroup_uniform_control_flow = true;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::MaterializeScalarMaskWord), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::LaneId), 1U);
    Shader::Optimization::CollectShaderInfoPass(program, profile);

    Shader::Backend::Bindings bindings{};
    const auto spirv =
        Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
    EXPECT_EQ(CountSpirvOpcode(spirv, spv::Op::OpGroupNonUniformBallot), 0U);
    EXPECT_TRUE(HasExecutionMode(spirv, static_cast<spv::ExecutionMode>(4421)));
}

TEST(SpirvLoopControl, PreservesDynamicScalarMaskLoopsForDriverCorrectness) {
    const auto emit_loop = [](bool materializes_dynamic_mask,
                              std::array<u32, 3> workgroup_size = {8, 8, 1}) {
        Shader::Info info{};
        info.stage = Shader::Stage::Compute;
        info.l_stage = Shader::LogicalStage::Compute;
        Shader::IR::Program program{info};
        Shader::Pools pools{};

        auto* const header = pools.block_pool.Create(pools.inst_pool);
        auto* const body = pools.block_pool.Create(pools.inst_pool);
        auto* const continue_block = pools.block_pool.Create(pools.inst_pool);
        auto* const merge = pools.block_pool.Create(pools.inst_pool);
        program.blocks = {header, body, continue_block, merge};

        Shader::IR::IREmitter header_ir{*header};
        header_ir.Prologue();
        Shader::IR::IREmitter body_ir{*body};
        if (materializes_dynamic_mask) {
            body_ir.Reference(
                body_ir.MaterializeScalarMaskWord(body_ir.Imm1(true), body_ir.Imm1(true),
                                                  body_ir.Imm32(0), 0));
        }
        Shader::IR::IREmitter continue_ir{*continue_block};
        const auto repeat = continue_ir.Imm1(false);
        Shader::IR::IREmitter merge_ir{*merge};
        merge_ir.Epilogue();

        program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
        program.syntax_list.back().data.block = header;
        auto& loop = program.syntax_list.emplace_back();
        loop.type = Shader::IR::AbstractSyntaxNode::Type::Loop;
        loop.data.loop = {.body = body, .continue_block = continue_block, .merge = merge};
        program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
        program.syntax_list.back().data.block = body;
        program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
        program.syntax_list.back().data.block = continue_block;
        auto& repeat_node = program.syntax_list.emplace_back();
        repeat_node.type = Shader::IR::AbstractSyntaxNode::Type::Repeat;
        repeat_node.data.repeat = {.cond = repeat, .loop_header = header, .merge = merge};
        program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
        program.syntax_list.back().data.block = merge;
        program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;
        Shader::RuntimeInfo runtime_info{};
        runtime_info.Initialize(Shader::Stage::Compute);
        runtime_info.cs_info.workgroup_size = workgroup_size;
        Shader::Profile profile{};
        profile.supported_spirv = 0x00010600;
        profile.subgroup_size = 32;
        Shader::Optimization::CollectShaderInfoPass(program, profile);
        Shader::Backend::Bindings bindings{};
        auto spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
        return std::pair{std::move(spirv), program.info.has_dynamic_scalar_mask_loop};
    };

    const auto [dynamic_spirv, dynamic_mask_loop] = emit_loop(true);
    EXPECT_TRUE(dynamic_mask_loop);
    const auto dynamic_control = FirstLoopControl(dynamic_spirv);
    ASSERT_TRUE(dynamic_control);
    EXPECT_EQ(dynamic_control->first, spv::LoopControlMaskNone);
    EXPECT_EQ(dynamic_control->second, 0U);

    const auto [native_subgroup_spirv, native_subgroup_mask_loop] =
        emit_loop(true, {32, 1, 1});
    EXPECT_TRUE(native_subgroup_mask_loop);
    const auto native_subgroup_control = FirstLoopControl(native_subgroup_spirv);
    ASSERT_TRUE(native_subgroup_control);
    EXPECT_EQ(native_subgroup_control->first, spv::LoopControlMaskNone);
    EXPECT_EQ(native_subgroup_control->second, 0U);

    const auto [ordinary_spirv, ordinary_mask_loop] = emit_loop(false);
    EXPECT_FALSE(ordinary_mask_loop);
    const auto ordinary_control = FirstLoopControl(ordinary_spirv);
    ASSERT_TRUE(ordinary_control);
    EXPECT_EQ(ordinary_control->first, spv::LoopControlMaskNone);
    EXPECT_EQ(ordinary_control->second, 0U);
}

TEST(Wave64SubgroupLoweringPass, LowersWriteLaneToGuestLaneSelect) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    Shader::IR::IREmitter ir{*block};
    ir.Reference(ir.WriteLane(ir.Imm32(3), ir.Imm32(7), ir.Imm32(42)));
    ir.Epilogue();
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::WriteLane), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::IEqual32), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::SelectU32), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::GetAttributeU32), 3U);
}

TEST(Wave64SubgroupLoweringPass, ZerosReadFromInactiveLaneInPartialGuestWave) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U32 read = ir.ReadLane(ir.Imm32(0x12345678), ir.Imm32(32));
    ir.Reference(ir.IAdd(read, ir.Imm32(1)));
    ir.Epilogue();
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {32, 1, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ReadLane), 0U);
}

TEST(Wave64SubgroupLoweringPass, PreservesWriteToInactiveLaneInPartialGuestWave) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    Shader::IR::IREmitter ir{*block};
    ir.Reference(ir.WriteLane(ir.Imm32(3), ir.Imm32(7), ir.Imm32(32)));
    ir.Epilogue();
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {32, 1, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::WriteLane), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::SelectU32), 0U);
}

TEST(Wave64SubgroupLoweringPass, SelectsWriteTargetInPartialGuestWave) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    Shader::IR::IREmitter ir{*block};
    ir.Reference(ir.WriteLane(ir.Imm32(3), ir.Imm32(7), ir.Imm32(5)));
    ir.Epilogue();
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {32, 1, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::WriteLane), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::IEqual32), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::SelectU32), 1U);
}

TEST(Wave64SubgroupLoweringPass, LowersUniformReadLaneThroughWorkgroupSharedMemory) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U32 lane = ir.LaneId();
    ir.Reference(ir.ReadLane(lane, ir.Imm32(32)));
    ir.Reference(ir.ShuffleXor(lane, ir.Imm32(16)));
    ir.Reference(ir.ReadFirstLane(lane));
    ir.Reference(ir.Ballot(ir.INotEqual(lane, ir.Imm32(0))));
    ir.Epilogue();
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;

    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::LaneId), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ReadLane), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ReadFirstLane), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64ReadFirstActiveLane), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ShuffleXor), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Ballot), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::GetAttributeU32), 9U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::WriteSharedU32), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::LoadSharedU32), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Barrier), 2U);
    EXPECT_EQ(runtime_info.cs_info.shared_memory_size, 256U);
}

TEST(Wave64SubgroupLoweringPass, DoesNotPublishWaveSplitWhenExchangeScratchCannotFit) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U32 lane = ir.LaneId();
    ir.Reference(ir.ReadLane(lane, ir.Imm32(32)));
    ir.Epilogue();
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    runtime_info.cs_info.shared_memory_size = 1024;
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_FALSE(program.info.emulate_wave64);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::LaneId), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ReadLane), 1U);
}

TEST(Wave64SubgroupLoweringPass, LeavesCrossHalfShuffleShaderUntouched) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U32 lane = ir.LaneId();
    ir.Reference(ir.ShuffleXor(lane, ir.Imm32(32)));
    ir.Epilogue();
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::LaneId), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ShuffleXor), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::GetAttributeU32), 0U);
}

TEST(Wave64SubgroupLoweringPass, LowersReadLaneAfterStructuredLoopReconverges) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const header = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const read_block = pools.block_pool.Create(pools.inst_pool);
    auto* const continue_block = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {header, body, read_block, continue_block, merge};
    Shader::IR::IREmitter header_ir{*header};
    const Shader::IR::U32 lane = header_ir.LaneId();
    Shader::IR::IREmitter continue_ir{*continue_block};
    const Shader::IR::U1 repeat = continue_ir.Imm1(false);
    Shader::IR::IREmitter merge_ir{*merge};
    merge_ir.Reference(merge_ir.ReadLane(lane, merge_ir.Imm32(32)));
    merge_ir.Epilogue();

    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = header;
    auto& loop = program.syntax_list.emplace_back();
    loop.type = Shader::IR::AbstractSyntaxNode::Type::Loop;
    loop.data.loop = {.body = body, .continue_block = continue_block, .merge = merge};
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = body;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = continue_block;
    auto& repeat_node = program.syntax_list.emplace_back();
    repeat_node.type = Shader::IR::AbstractSyntaxNode::Type::Repeat;
    repeat_node.data.repeat = {.cond = repeat, .loop_header = header, .merge = merge};
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = merge;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::LaneId), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ReadLane), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::WriteSharedU32), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::LoadSharedU32), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Barrier), 2U);
}

TEST(Wave64SubgroupLoweringPass,
     DeclinesSequentialLaneRetirementWithSplitScalarMaskState) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    const auto make_block = [&] {
        auto* const block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        return block;
    };
    auto* const entry = make_block();
    auto* const outer_body = make_block();
    auto* const loop_header = make_block();
    auto* const loop_body = make_block();
    auto* const read_block = make_block();
    auto* const loop_continue = make_block();
    auto* const loop_merge = make_block();
    auto* const second_loop_header = make_block();
    auto* const second_loop_body = make_block();
    auto* const second_read_block = make_block();
    auto* const second_loop_continue = make_block();
    auto* const second_loop_merge = make_block();
    auto* const outer_merge = make_block();

    Shader::IR::IREmitter entry_ir{*entry};
    const Shader::IR::U32 invocation =
        entry_ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U1 outer_cond = entry_ir.ConditionRef(
        entry_ir.IGreaterThanEqual(invocation, entry_ir.Imm32(5U), false));
    entry_ir.Reference(
        entry_ir.MaterializeScalarMaskWord(entry_ir.Imm1(true), outer_cond,
                                           entry_ir.Imm32(0U), 0U));
    Shader::IR::IREmitter{*outer_body}.Epilogue();
    Shader::IR::IREmitter{*loop_header}.Epilogue();

    Shader::IR::IREmitter body_ir{*loop_body};
    const Shader::IR::U1 lane_continues =
        body_ir.IGreaterThanEqual(invocation, body_ir.Imm32(10U), false);
    const Shader::IR::U1 break_cond =
        body_ir.ConditionRef(body_ir.LogicalNot(lane_continues));

    Shader::IR::IREmitter read_ir{*read_block};
    const Shader::IR::U32 first_active_lane = read_ir.ReadFirstLane(invocation);
    read_ir.Reference(first_active_lane);
    read_ir.Epilogue();

    Shader::IR::IREmitter continue_ir{*loop_continue};
    const Shader::IR::U1 repeat_cond = continue_ir.ConditionRef(continue_ir.Imm1(true));
    continue_ir.Epilogue();
    Shader::IR::IREmitter{*loop_merge}.Epilogue();
    Shader::IR::IREmitter{*second_loop_header}.Epilogue();
    Shader::IR::IREmitter second_body_ir{*second_loop_body};
    const Shader::IR::U1 second_lane_continues =
        second_body_ir.IGreaterThanEqual(invocation, second_body_ir.Imm32(20U), false);
    const Shader::IR::U1 second_break_cond =
        second_body_ir.ConditionRef(second_body_ir.LogicalNot(second_lane_continues));
    Shader::IR::IREmitter second_read_ir{*second_read_block};
    const Shader::IR::U32 second_first_active_lane =
        second_read_ir.ReadFirstLane(invocation);
    second_read_ir.Reference(second_first_active_lane);
    second_read_ir.Epilogue();
    Shader::IR::IREmitter second_continue_ir{*second_loop_continue};
    const Shader::IR::U1 second_repeat_cond =
        second_continue_ir.ConditionRef(second_continue_ir.Imm1(true));
    second_continue_ir.Epilogue();
    Shader::IR::IREmitter{*second_loop_merge}.Epilogue();
    Shader::IR::IREmitter{*outer_merge}.Epilogue();

    const auto add_block = [&](Shader::IR::Block* block) {
        auto& node = program.syntax_list.emplace_back();
        node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
    };
    add_block(entry);
    auto& outer_if = program.syntax_list.emplace_back();
    outer_if.type = Shader::IR::AbstractSyntaxNode::Type::If;
    outer_if.data.if_node = {.cond = outer_cond, .body = outer_body, .merge = outer_merge};
    add_block(outer_body);
    add_block(loop_header);
    auto& loop = program.syntax_list.emplace_back();
    loop.type = Shader::IR::AbstractSyntaxNode::Type::Loop;
    loop.data.loop = {
        .body = loop_body, .continue_block = loop_continue, .merge = loop_merge};
    add_block(loop_body);
    auto& break_node = program.syntax_list.emplace_back();
    break_node.type = Shader::IR::AbstractSyntaxNode::Type::Break;
    break_node.data.break_node = {
        .cond = break_cond, .merge = loop_merge, .skip = read_block};
    add_block(read_block);
    add_block(loop_continue);
    auto& repeat = program.syntax_list.emplace_back();
    repeat.type = Shader::IR::AbstractSyntaxNode::Type::Repeat;
    repeat.data.repeat = {
        .cond = repeat_cond, .loop_header = loop_header, .merge = loop_merge};
    add_block(loop_merge);
    add_block(second_loop_header);
    auto& second_loop = program.syntax_list.emplace_back();
    second_loop.type = Shader::IR::AbstractSyntaxNode::Type::Loop;
    second_loop.data.loop = {.body = second_loop_body,
                             .continue_block = second_loop_continue,
                             .merge = second_loop_merge};
    add_block(second_loop_body);
    auto& second_break_node = program.syntax_list.emplace_back();
    second_break_node.type = Shader::IR::AbstractSyntaxNode::Type::Break;
    second_break_node.data.break_node = {
        .cond = second_break_cond, .merge = second_loop_merge, .skip = second_read_block};
    add_block(second_read_block);
    add_block(second_loop_continue);
    auto& second_repeat = program.syntax_list.emplace_back();
    second_repeat.type = Shader::IR::AbstractSyntaxNode::Type::Repeat;
    second_repeat.data.repeat = {.cond = second_repeat_cond,
                                 .loop_header = second_loop_header,
                                 .merge = second_loop_merge};
    add_block(second_loop_merge);
    auto& outer_end = program.syntax_list.emplace_back();
    outer_end.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
    outer_end.data.end_if.merge = outer_merge;
    add_block(outer_merge);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ReadFirstLane), 2U);
    EXPECT_EQ(runtime_info.cs_info.shared_memory_size, 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::MaterializeScalarMaskWord), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64MaterializeScalarMaskWord), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64LaneRetirementInit), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64LaneRetirementSync), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64ReadFirstLane), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64GroupAny), 0U);
}

TEST(Wave64SubgroupLoweringPass, DeclinesControlStateInitializedInsideTopLevelLoop) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const header = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const read_block = pools.block_pool.Create(pools.inst_pool);
    auto* const continue_block = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {header, body, read_block, continue_block, merge};

    Shader::IR::IREmitter header_ir{*header};
    const Shader::IR::U32 invocation =
        header_ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    Shader::IR::IREmitter body_ir{*body};
    const Shader::IR::U1 break_cond = body_ir.ConditionRef(body_ir.Imm1(false));
    Shader::IR::IREmitter read_ir{*read_block};
    read_ir.Reference(read_ir.ReadFirstLane(invocation));
    Shader::IR::IREmitter continue_ir{*continue_block};
    const Shader::IR::U1 repeat = continue_ir.ConditionRef(continue_ir.Imm1(false));
    Shader::IR::IREmitter{*merge}.Epilogue();

    const auto add_block = [&](Shader::IR::Block* block) {
        auto& node = program.syntax_list.emplace_back();
        node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
    };
    add_block(header);
    auto& loop = program.syntax_list.emplace_back();
    loop.type = Shader::IR::AbstractSyntaxNode::Type::Loop;
    loop.data.loop = {.body = body, .continue_block = continue_block, .merge = merge};
    add_block(body);
    auto& break_node = program.syntax_list.emplace_back();
    break_node.type = Shader::IR::AbstractSyntaxNode::Type::Break;
    break_node.data.break_node = {.cond = break_cond, .merge = merge, .skip = read_block};
    add_block(read_block);
    add_block(continue_block);
    auto& repeat_node = program.syntax_list.emplace_back();
    repeat_node.type = Shader::IR::AbstractSyntaxNode::Type::Repeat;
    repeat_node.data.repeat = {.cond = repeat, .loop_header = header, .merge = merge};
    add_block(merge);

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ReadFirstLane), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64ReadFirstLane), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64LaneRetirementInit), 0U);
    EXPECT_EQ(runtime_info.cs_info.shared_memory_size, 0U);
}

TEST(Wave64SubgroupLoweringPass, DeclinesOperationBeforeLaterExitInSameControlRegion) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const entry = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {entry, body, merge};
    Shader::IR::IREmitter entry_ir{*entry};
    const Shader::IR::U32 invocation =
        entry_ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U1 condition =
        entry_ir.ConditionRef(entry_ir.IGreaterThanEqual(invocation, entry_ir.Imm32(32U), false));
    Shader::IR::IREmitter body_ir{*body};
    body_ir.Reference(body_ir.ReadFirstLane(invocation));
    Shader::IR::IREmitter{*merge}.Epilogue();

    const auto add_block = [&](Shader::IR::Block* block) {
        auto& node = program.syntax_list.emplace_back();
        node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
    };
    add_block(entry);
    auto& if_node = program.syntax_list.emplace_back();
    if_node.type = Shader::IR::AbstractSyntaxNode::Type::If;
    if_node.data.if_node = {.cond = condition, .body = body, .merge = merge};
    add_block(body);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;
    auto& end_if = program.syntax_list.emplace_back();
    end_if.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
    end_if.data.end_if.merge = merge;
    add_block(merge);

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ReadFirstLane), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64ReadFirstLane), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64LaneRetirementInit), 0U);
    EXPECT_EQ(runtime_info.cs_info.shared_memory_size, 0U);
}

TEST(Wave64SubgroupLoweringPass, DeclinesMixedControlledAndBarrierWaveExchanges) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const entry = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {entry, body, merge};
    Shader::IR::IREmitter entry_ir{*entry};
    const Shader::IR::U32 invocation =
        entry_ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U1 condition =
        entry_ir.ConditionRef(entry_ir.IGreaterThanEqual(invocation, entry_ir.Imm32(32U), false));
    Shader::IR::IREmitter body_ir{*body};
    body_ir.Reference(body_ir.ReadFirstLane(invocation));
    Shader::IR::IREmitter merge_ir{*merge};
    merge_ir.Reference(merge_ir.ReadLane(invocation, merge_ir.Imm32(32U)));
    merge_ir.Epilogue();

    const auto add_block = [&](Shader::IR::Block* block) {
        auto& node = program.syntax_list.emplace_back();
        node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
    };
    add_block(entry);
    auto& if_node = program.syntax_list.emplace_back();
    if_node.type = Shader::IR::AbstractSyntaxNode::Type::If;
    if_node.data.if_node = {.cond = condition, .body = body, .merge = merge};
    add_block(body);
    auto& end_if = program.syntax_list.emplace_back();
    end_if.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
    end_if.data.end_if.merge = merge;
    add_block(merge);

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ReadFirstLane), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ReadLane), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64ReadFirstLane), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Barrier), 0U);
    EXPECT_EQ(runtime_info.cs_info.shared_memory_size, 0U);
}

TEST(Wave64SubgroupLoweringPass, RemovesDeadWave64ExchangeAndItsControlProtocol) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};
    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U32 invocation =
        ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    ir.Wave64LaneRetirementInit(ir.Imm1(true), invocation, ir.Imm32(64U));
    ir.Wave64LaneRetirementSync(ir.Imm1(true), invocation, ir.Imm32(64U));
    static_cast<void>(ir.Wave64MaterializeScalarMaskWord(
        ir.Imm1(true), ir.Imm1(true), ir.Imm32(0U), 0U, invocation, ir.Imm32(64U)));
    ir.Epilogue();

    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::Wave64ProtocolCleanupPass(program);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64MaterializeScalarMaskWord), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64LaneRetirementInit), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64LaneRetirementSync), 0U);
}

TEST(Wave64SubgroupLoweringPass, RetainsLiveReadFirstLaneControlProtocol) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};
    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U32 invocation =
        ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    ir.Wave64LaneRetirementInit(ir.Imm1(true), invocation, ir.Imm32(64U));
    ir.Wave64LaneRetirementSync(ir.Imm1(true), invocation, ir.Imm32(64U));
    ir.Reference(ir.Wave64ReadFirstLane(ir.Imm32(9U), ir.Imm1(true), invocation,
                                        ir.Imm32(64U)));
    ir.Epilogue();

    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::Wave64ProtocolCleanupPass(program);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64ReadFirstLane), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64LaneRetirementInit), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Wave64LaneRetirementSync), 1U);
}

TEST(Wave64SubgroupLoweringPass, LeavesDivergentReadLaneShaderUntouched) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const condition_block = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {condition_block, body, merge};
    Shader::IR::IREmitter condition_ir{*condition_block};
    const Shader::IR::U32 lane = condition_ir.LaneId();
    const Shader::IR::U1 condition = condition_ir.INotEqual(lane, condition_ir.Imm32(0));
    Shader::IR::IREmitter body_ir{*body};
    body_ir.Reference(body_ir.ReadLane(lane, body_ir.Imm32(32)));
    Shader::IR::IREmitter merge_ir{*merge};
    merge_ir.Epilogue();

    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = condition_block;
    auto& if_node = program.syntax_list.emplace_back();
    if_node.type = Shader::IR::AbstractSyntaxNode::Type::If;
    if_node.data.if_node = {.cond = condition, .body = body, .merge = merge};
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = body;
    auto& end_if = program.syntax_list.emplace_back();
    end_if.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
    end_if.data.end_if.merge = merge;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = merge;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::LaneId), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::ReadLane), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::WriteSharedU32), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Barrier), 0U);
}

TEST(Wave64SubgroupLoweringPass, DivergentSplitScalarMaskUsesGuestWaveLaneId) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const condition_block = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {condition_block, body, merge};
    Shader::IR::IREmitter condition_ir{*condition_block};
    const Shader::IR::U32 lane = condition_ir.LaneId();
    const Shader::IR::U1 condition = condition_ir.INotEqual(lane, condition_ir.Imm32(0));
    Shader::IR::IREmitter body_ir{*body};
    body_ir.Reference(body_ir.MaterializeScalarMaskWord(body_ir.Imm1(true), body_ir.Imm1(true),
                                                        body_ir.Imm32(0), 1));
    Shader::IR::IREmitter merge_ir{*merge};
    merge_ir.Epilogue();

    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = condition_block;
    auto& if_node = program.syntax_list.emplace_back();
    if_node.type = Shader::IR::AbstractSyntaxNode::Type::If;
    if_node.data.if_node = {.cond = condition, .body = body, .merge = merge};
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = body;
    auto& end_if = program.syntax_list.emplace_back();
    end_if.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
    end_if.data.end_if.merge = merge;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = merge;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::LaneId), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::GetAttributeU32), 3U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::MaterializeScalarMaskWord), 1U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::WriteSharedU32), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Barrier), 0U);
}

TEST(Wave64SubgroupLoweringPass, IgnoresEliminatedDivergentReadLane) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const entry = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {entry, body, merge};
    Shader::IR::IREmitter entry_ir{*entry};
    const Shader::IR::U32 lane = entry_ir.LaneId();
    entry_ir.Reference(lane);
    const Shader::IR::U32 written =
        entry_ir.WriteLane(entry_ir.Imm32(3), entry_ir.Imm32(7), entry_ir.Imm32(42));
    const Shader::IR::U1 condition = entry_ir.INotEqual(lane, entry_ir.Imm32(0));
    Shader::IR::IREmitter body_ir{*body};
    body_ir.Reference(body_ir.ReadLane(written, body_ir.Imm32(42)));
    Shader::IR::IREmitter merge_ir{*merge};
    merge_ir.Epilogue();

    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = entry;
    auto& if_node = program.syntax_list.emplace_back();
    if_node.type = Shader::IR::AbstractSyntaxNode::Type::If;
    if_node.data.if_node = {.cond = condition, .body = body, .merge = merge};
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = body;
    auto& end_if = program.syntax_list.emplace_back();
    end_if.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
    end_if.data.end_if.merge = merge;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = merge;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::ReadLaneEliminationPass(program);
    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::LaneId), 0U);
    // One flattened index replaces LaneId and another lowers the still-live WriteLane.
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::GetAttributeU32), 6U);
}

TEST(SharedMemoryBarrierPass, LaneDivergenceSynchronizesTheMergeBeforeLdsReads) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const condition_block = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter condition_ir{*condition_block};
    const Shader::IR::U1 condition =
        condition_ir.INotEqual(condition_ir.LaneId(), condition_ir.Imm32(0));
    Shader::IR::IREmitter body_ir{*body};
    body_ir.WriteShared(32, body_ir.Imm32(7), body_ir.Imm32(0));
    Shader::IR::IREmitter merge_ir{*merge};
    merge_ir.Reference(merge_ir.LoadShared(32, false, merge_ir.Imm32(0)));
    merge_ir.Epilogue();

    auto& if_node = program.syntax_list.emplace_back();
    if_node.type = Shader::IR::AbstractSyntaxNode::Type::If;
    if_node.data.if_node = {.cond = condition, .body = body, .merge = merge};
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = body;
    auto& end_if = program.syntax_list.emplace_back();
    end_if.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
    end_if.data.end_if.merge = merge;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = merge;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.shared_memory_size = 256;
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.needs_lds_barriers = true;

    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);

    const auto first_non_phi = std::ranges::find_if_not(merge->Instructions(), Shader::IR::IsPhi);
    ASSERT_NE(first_non_phi, merge->Instructions().end());
    EXPECT_EQ(first_non_phi->GetOpcode(), Shader::IR::Opcode::Barrier);
}

TEST(SharedMemoryBarrierPass, SubWaveThreadgroupOrdersStraightLineLdsWriteThenRead) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    Shader::IR::IREmitter ir{*block};
    ir.WriteShared(32, ir.Imm32(7), ir.Imm32(0));
    ir.Reference(ir.LoadShared(32, false, ir.Imm32(0)));
    ir.Epilogue();
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.shared_memory_size = 256;
    runtime_info.cs_info.workgroup_size = {16, 1, 1};
    Shader::Profile profile{};
    profile.needs_lds_barriers = true;

    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);

    const auto barrier = std::ranges::find_if(block->Instructions(), [](const auto& inst) {
        return inst.GetOpcode() == Shader::IR::Opcode::Barrier;
    });
    EXPECT_NE(barrier, block->Instructions().end());
}

} // namespace
