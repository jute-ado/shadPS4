// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <gtest/gtest.h>

#include "common/object_pool.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/backend/spirv/read_lane_policy.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/recompiler.h"
#include "spirv/unified1/spirv.hpp"

namespace {

using namespace Shader;
using Shader::Backend::SPIRV::ShouldUseZeroForInactiveReadLane;

std::vector<u32> EmitLaneIdShader(LogicalStage logical_stage,
                                  bool supports_maximal_reconvergence = false,
                                  u64 pgm_hash = 0,
                                  bool supports_subgroup_uniform_control_flow = false) {
    Info info{};
    info.stage = logical_stage == LogicalStage::Compute
                     ? Stage::Compute
                     : logical_stage == LogicalStage::Vertex ? Stage::Vertex : Stage::Fragment;
    info.l_stage = logical_stage;
    info.pgm_hash = pgm_hash;
    info.uses_lane_id = true;

    IR::Program program{info};
    Pools pools{};
    IR::Block* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    IR::IREmitter ir{*block};
    ir.Prologue();
    ir.Reference(ir.LaneId());
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.supports_maximal_reconvergence = supports_maximal_reconvergence;
    profile.supports_subgroup_uniform_control_flow = supports_subgroup_uniform_control_flow;

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(info.stage);
    runtime_info.cs_info.workgroup_size = {64, 1, 1};
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

std::vector<u32> EmitReadLaneShader(u32 subgroup_size, u32 workgroup_size, u32 lane) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.uses_lane_id = true;
    info.uses_read_lane = true;
    IR::Program program{info};
    Pools pools{};
    IR::Block* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    IR::IREmitter ir{*block};
    ir.Prologue();
    ir.Reference(ir.ReadLane(ir.Imm32(0x12345678), ir.Imm32(lane)));
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = subgroup_size;
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(info.stage);
    runtime_info.cs_info.workgroup_size = {workgroup_size, 1, 1};
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

std::vector<u32> EmitShuffleXorShader(u32 mask) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    IR::Program program{info};
    Pools pools{};
    IR::Block* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    IR::IREmitter ir{*block};
    ir.Prologue();
    ir.Reference(ir.ShuffleXor(ir.Imm32(0x12345678), ir.Imm32(mask)));
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(info.stage);
    runtime_info.cs_info.workgroup_size = {32, 1, 1};
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

std::vector<u32> EmitBallotShader() {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    IR::Program program{info};
    Pools pools{};
    IR::Block* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    IR::IREmitter ir{*block};
    ir.Prologue();
    ir.Reference(ir.Ballot(ir.Imm1(true)));
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    Optimization::CollectShaderInfoPass(program, profile);
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(info.stage);
    runtime_info.cs_info.workgroup_size = {32, 1, 1};
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

std::vector<u32> EmitConditionalBallotWordShader() {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    IR::Program program{info};
    Pools pools{};
    IR::Block* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    IR::IREmitter ir{*block};
    ir.Prologue();
    const IR::U1 dynamic_condition = ir.INotEqual(ir.LaneId(), ir.Imm32(0));
    ir.Reference(ir.MaterializeScalarMaskWord(dynamic_condition, ir.Imm1(true),
                                              ir.Imm32(0x12345678), 1));
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    Optimization::CollectShaderInfoPass(program, profile);
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(info.stage);
    runtime_info.cs_info.workgroup_size = {32, 1, 1};
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

bool HasOpcode(const std::vector<u32>& spirv, spv::Op opcode) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == opcode) {
            return true;
        }
        EXPECT_GT(words, 0u);
        offset += words;
    }
    return false;
}

bool HasBuiltIn(const std::vector<u32>& spirv, spv::BuiltIn builtin) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpDecorate && words >= 4 &&
            spirv[offset + 2] == static_cast<u32>(spv::DecorationBuiltIn) &&
            spirv[offset + 3] == static_cast<u32>(builtin)) {
            return true;
        }
        EXPECT_GT(words, 0u);
        offset += words;
    }
    return false;
}

bool BuiltInHasDecoration(const std::vector<u32>& spirv, spv::BuiltIn builtin,
                          spv::Decoration decoration) {
    u32 variable{};
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpDecorate && words >= 4 &&
            spirv[offset + 2] == static_cast<u32>(spv::DecorationBuiltIn) &&
            spirv[offset + 3] == static_cast<u32>(builtin)) {
            variable = spirv[offset + 1];
            break;
        }
        EXPECT_GT(words, 0u);
        offset += words;
    }
    if (variable == 0) {
        return false;
    }
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpDecorate && words >= 3 &&
            spirv[offset + 1] == variable &&
            spirv[offset + 2] == static_cast<u32>(decoration)) {
            return true;
        }
        EXPECT_GT(words, 0u);
        offset += words;
    }
    return false;
}

bool HasExecutionMode(const std::vector<u32>& spirv, spv::ExecutionMode mode) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpExecutionMode && words >= 3 &&
            spirv[offset + 2] == static_cast<u32>(mode)) {
            return true;
        }
        EXPECT_GT(words, 0u);
        offset += words;
    }
    return false;
}

bool HasCapability(const std::vector<u32>& spirv, spv::Capability capability) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpCapability && words >= 2 &&
            spirv[offset + 1] == static_cast<u32>(capability)) {
            return true;
        }
        EXPECT_GT(words, 0u);
        offset += words;
    }
    return false;
}

bool HasIrOpcode(const IR::Block& block, IR::Opcode opcode) {
    return std::ranges::any_of(block.Instructions(),
                               [opcode](const IR::Inst& inst) { return inst.GetOpcode() == opcode; });
}

TEST(LaneIdSpirv, ComputeUsesWorkgroupLocalIndexForEmulatedWave64Lane) {
    const auto spirv = EmitLaneIdShader(LogicalStage::Compute);

    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInLocalInvocationIndex));
    EXPECT_FALSE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
}

TEST(LaneIdSpirv, NonComputeKeepsNativeSubgroupLane) {
    const auto spirv = EmitLaneIdShader(LogicalStage::Fragment);

    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
    EXPECT_FALSE(HasBuiltIn(spirv, spv::BuiltInLocalInvocationIndex));
}

TEST(LaneIdSpirv, VertexSubgroupLaneBuiltinIsNotFlatDecorated) {
    const auto spirv = EmitLaneIdShader(LogicalStage::Vertex);

    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
    EXPECT_FALSE(BuiltInHasDecoration(spirv, spv::BuiltInSubgroupLocalInvocationId,
                                     spv::DecorationFlat));
}

TEST(LaneIdSpirv, FragmentSubgroupLaneBuiltinIsFlatDecorated) {
    const auto spirv = EmitLaneIdShader(LogicalStage::Fragment);

    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
    EXPECT_TRUE(BuiltInHasDecoration(spirv, spv::BuiltInSubgroupLocalInvocationId,
                                     spv::DecorationFlat));
}

TEST(LaneIdSpirv, ComputeGroupOperationsRequestSubgroupUniformControlFlowWhenSupported) {
    constexpr auto SubgroupUniformControlFlow =
        static_cast<spv::ExecutionMode>(spv::ExecutionModeSubgroupUniformControlFlowKHR);
    const auto supported = EmitLaneIdShader(LogicalStage::Compute, false, 0, true);
    const auto unsupported = EmitLaneIdShader(LogicalStage::Compute);

    EXPECT_TRUE(HasExecutionMode(supported, SubgroupUniformControlFlow));
    EXPECT_FALSE(HasExecutionMode(unsupported, SubgroupUniformControlFlow));
}

TEST(LaneIdSpirv, InactiveGuestLaneFallsBackToCurrentInvocationValue) {
    const auto spirv = EmitReadLaneShader(32, 32, 32);

    EXPECT_FALSE(HasOpcode(spirv, spv::Op::OpGroupNonUniformShuffle));
    EXPECT_TRUE(HasOpcode(spirv, spv::Op::OpGroupNonUniformBroadcast));
    EXPECT_TRUE(HasOpcode(spirv, spv::Op::OpBitwiseAnd));
    EXPECT_TRUE(HasOpcode(spirv, spv::Op::OpULessThan));
    EXPECT_TRUE(HasOpcode(spirv, spv::Op::OpSelect));
}

TEST(LaneIdSpirv, ZeroFallbackDiagnosticTargetsOnlyTheObservedShadowKernel) {
    EXPECT_TRUE(ShouldUseZeroForInactiveReadLane(0x87cf9b8a, true));
    EXPECT_FALSE(ShouldUseZeroForInactiveReadLane(0x87cf9b8a, false));
    EXPECT_FALSE(ShouldUseZeroForInactiveReadLane(0x12345678, true));
}

TEST(LaneIdSpirv, LargerGuestWaveUsesWorkgroupExchangeForCrossSubgroupLane) {
    const auto spirv = EmitReadLaneShader(32, 64, 32);

    EXPECT_FALSE(HasOpcode(spirv, spv::Op::OpGroupNonUniformShuffle));
    EXPECT_FALSE(HasOpcode(spirv, spv::Op::OpGroupNonUniformBroadcast));
    EXPECT_TRUE(HasOpcode(spirv, spv::Op::OpControlBarrier));
    EXPECT_TRUE(HasOpcode(spirv, spv::Op::OpStore));
    EXPECT_TRUE(HasOpcode(spirv, spv::Op::OpLoad));
    EXPECT_TRUE(HasOpcode(spirv, spv::Op::OpSelect));
}

TEST(LaneIdSpirv, PerInvocationXorLaneUsesShuffleInsteadOfBroadcast) {
    const auto spirv = EmitShuffleXorShader(16);

    EXPECT_TRUE(HasOpcode(spirv, spv::Op::OpGroupNonUniformShuffleXor));
    EXPECT_FALSE(HasOpcode(spirv, spv::Op::OpGroupNonUniformBroadcast));
}

TEST(LaneIdSpirv, BallotDeclaresGroupNonUniformBallotCapability) {
    const auto spirv = EmitBallotShader();

    EXPECT_TRUE(HasOpcode(spirv, spv::Op::OpGroupNonUniformBallot));
    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityGroupNonUniformBallot));
}

TEST(LaneIdSpirv, ConditionalMaskMaterializationBranchesAroundUnusedBallot) {
    const auto spirv = EmitConditionalBallotWordShader();

    EXPECT_TRUE(HasOpcode(spirv, spv::Op::OpBranchConditional));
    EXPECT_TRUE(HasOpcode(spirv, spv::Op::OpGroupNonUniformBallot));
    EXPECT_TRUE(HasOpcode(spirv, spv::Op::OpPhi));
}

TEST(LaneIdSpirv, UnwrittenScalarDefaultsToScalarRepresentation) {
    Pools pools{};
    IR::Block* block = pools.block_pool.Create(pools.inst_pool);
    IR::IREmitter ir{*block};
    ir.Reference(ir.GetScalarReg(IR::ScalarReg::S0));
    IR::BlockList blocks{block};

    Optimization::SsaRewritePass(blocks);

    EXPECT_FALSE(HasIrOpcode(*block, IR::Opcode::MaterializeScalarMaskWord));
    EXPECT_FALSE(HasIrOpcode(*block, IR::Opcode::Ballot));
}

} // namespace
