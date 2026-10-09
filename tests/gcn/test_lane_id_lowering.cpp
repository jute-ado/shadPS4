// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#endif

#include <gtest/gtest.h>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "spirv/unified1/spirv.hpp"

namespace {

using namespace Shader;

struct LaneIdPassFixture {
    explicit LaneIdPassFixture(std::array<u32, 3> workgroup_size = {4, 2, 8},
                               u32 subgroup_size = 32, Stage physical_stage = Stage::Compute)
        : program{info} {
        info.stage = physical_stage;
        info.l_stage = LogicalStage::Compute;
        block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        runtime_info.Initialize(Stage::Compute);
        runtime_info.cs_info.workgroup_size = workgroup_size;
        profile.subgroup_size = subgroup_size;
        profile.supported_spirv = 0x00010600;
    }

    Info info{};
    IR::Program program;
    Pools pools{};
    IR::Block* block{};
    RuntimeInfo runtime_info{};
    Profile profile{};
};

bool HasOpcode(const IR::Block& block, IR::Opcode opcode) {
    return std::ranges::any_of(
        block, [opcode](const IR::Inst& inst) { return inst.GetOpcode() == opcode; });
}

size_t CountOpcode(const IR::Block& block, IR::Opcode opcode) {
    return std::ranges::count_if(
        block, [opcode](const IR::Inst& inst) { return inst.GetOpcode() == opcode; });
}

void SetStraightLineSyntax(LaneIdPassFixture& fixture) {
    fixture.program.syntax_list.clear();
    fixture.program.syntax_list.emplace_back();
    fixture.program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    fixture.program.syntax_list.back().data.block = fixture.block;
    fixture.program.syntax_list.emplace_back();
    fixture.program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
}

void AppendBlockNode(IR::Program& program, IR::Block* block) {
    auto& node = program.syntax_list.emplace_back();
    node.type = IR::AbstractSyntaxNode::Type::Block;
    node.data.block = block;
}

void AppendIfNode(IR::Program& program, const IR::U1& condition, IR::Block* body,
                  IR::Block* merge) {
    auto& node = program.syntax_list.emplace_back();
    node.type = IR::AbstractSyntaxNode::Type::If;
    node.data.if_node.cond = condition;
    node.data.if_node.body = body;
    node.data.if_node.merge = merge;
}

void AppendEndIfNode(IR::Program& program, IR::Block* merge) {
    auto& node = program.syntax_list.emplace_back();
    node.type = IR::AbstractSyntaxNode::Type::EndIf;
    node.data.end_if.merge = merge;
}

bool ProgramHasOpcode(const IR::Program& program, IR::Opcode opcode) {
    return std::ranges::any_of(
        program.blocks, [opcode](const IR::Block* block) { return HasOpcode(*block, opcode); });
}

struct BracketedMergeBlocks {
    IR::Block* first_body{};
    IR::Block* merge{};
    IR::Block* second_body{};
    IR::Block* exit{};
};

BracketedMergeBlocks SetTwoSelectionBracketedMergeSyntax(LaneIdPassFixture& fixture,
                                                         const IR::U1& first_condition,
                                                         const IR::U1& second_condition) {
    auto* const first_body = fixture.pools.block_pool.Create(fixture.pools.inst_pool);
    auto* const merge = fixture.pools.block_pool.Create(fixture.pools.inst_pool);
    auto* const second_body = fixture.pools.block_pool.Create(fixture.pools.inst_pool);
    auto* const exit = fixture.pools.block_pool.Create(fixture.pools.inst_pool);

    fixture.block->AddBranch(first_body);
    fixture.block->AddBranch(merge);
    first_body->AddBranch(merge);
    merge->AddBranch(second_body);
    merge->AddBranch(exit);
    second_body->AddBranch(exit);
    fixture.program.blocks = {fixture.block, first_body, merge, second_body, exit};

    AppendBlockNode(fixture.program, fixture.block);
    AppendIfNode(fixture.program, first_condition, first_body, merge);
    AppendBlockNode(fixture.program, first_body);
    AppendEndIfNode(fixture.program, merge);
    AppendBlockNode(fixture.program, merge);
    AppendIfNode(fixture.program, second_condition, second_body, exit);
    AppendBlockNode(fixture.program, second_body);
    AppendEndIfNode(fixture.program, exit);
    AppendBlockNode(fixture.program, exit);
    fixture.program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    return {first_body, merge, second_body, exit};
}

IR::Inst* FindOpcode(IR::Block& block, IR::Opcode opcode) {
    const auto iter = std::ranges::find_if(
        block, [opcode](const IR::Inst& inst) { return inst.GetOpcode() == opcode; });
    return iter == block.end() ? nullptr : &*iter;
}

void ExpectLocalInvocationComponent(const IR::Inst& inst, u32 component) {
    ASSERT_EQ(inst.GetOpcode(), IR::Opcode::GetAttributeU32);
    EXPECT_EQ(inst.Arg(0).Attribute(), IR::Attribute::LocalInvocationId);
    EXPECT_EQ(inst.Arg(1).U32(), component);
}

TEST(LowerLaneIdToLocalInvocationIndexPass, FlattensMultidimensionalWave64Threadgroup) {
    LaneIdPassFixture fixture;
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.LaneId());

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::GetAttributeU32), 3U);

    const IR::Inst* const reference = FindOpcode(*fixture.block, IR::Opcode::Reference);
    ASSERT_NE(reference, nullptr);
    const IR::Inst* const root_add = reference->Arg(0).InstRecursive();
    ASSERT_EQ(root_add->GetOpcode(), IR::Opcode::IAdd32);

    const IR::Inst* const xy_add = root_add->Arg(0).InstRecursive();
    const IR::Inst* const z_stride = root_add->Arg(1).InstRecursive();
    ASSERT_EQ(xy_add->GetOpcode(), IR::Opcode::IAdd32);
    ASSERT_EQ(z_stride->GetOpcode(), IR::Opcode::IMul32);

    const IR::Inst* const x = xy_add->Arg(0).InstRecursive();
    const IR::Inst* const y_stride = xy_add->Arg(1).InstRecursive();
    ASSERT_EQ(y_stride->GetOpcode(), IR::Opcode::IMul32);
    ExpectLocalInvocationComponent(*x, 0);
    ExpectLocalInvocationComponent(*y_stride->Arg(0).InstRecursive(), 1);
    ExpectLocalInvocationComponent(*z_stride->Arg(0).InstRecursive(), 2);
    EXPECT_EQ(y_stride->Arg(1).U32(), 4U);
    EXPECT_EQ(z_stride->Arg(1).U32(), 8U);
}

enum class CrossLaneBlocker {
    WarpId,
    QuadShuffle,
    ReadFirstLane,
    ReadLane,
    WriteLane,
    Ballot,
    BallotFindLsb,
    GroupAny,
};

const char* BlockerName(CrossLaneBlocker blocker) {
    switch (blocker) {
    case CrossLaneBlocker::WarpId:
        return "WarpId";
    case CrossLaneBlocker::QuadShuffle:
        return "QuadShuffle";
    case CrossLaneBlocker::ReadFirstLane:
        return "ReadFirstLane";
    case CrossLaneBlocker::ReadLane:
        return "ReadLane";
    case CrossLaneBlocker::WriteLane:
        return "WriteLane";
    case CrossLaneBlocker::Ballot:
        return "Ballot";
    case CrossLaneBlocker::BallotFindLsb:
        return "BallotFindLsb";
    case CrossLaneBlocker::GroupAny:
        return "GroupAny";
    }
    return "Unknown";
}

void EmitLiveBlocker(IR::IREmitter& ir, CrossLaneBlocker blocker) {
    switch (blocker) {
    case CrossLaneBlocker::WarpId:
        ir.Reference(ir.WarpId());
        break;
    case CrossLaneBlocker::QuadShuffle:
        ir.Reference(ir.QuadShuffle(ir.Imm32(1), ir.Imm32(0)));
        break;
    case CrossLaneBlocker::ReadFirstLane:
        ir.Reference(ir.ReadFirstLane(ir.Imm32(1)));
        break;
    case CrossLaneBlocker::ReadLane:
        ir.Reference(ir.ReadLane(ir.Imm32(1), ir.Imm32(0)));
        break;
    case CrossLaneBlocker::WriteLane:
        ir.Reference(ir.WriteLane(ir.Imm32(1), ir.Imm32(2), ir.Imm32(0)));
        break;
    case CrossLaneBlocker::Ballot:
        ir.Reference(ir.Ballot(ir.Imm1(true)));
        break;
    case CrossLaneBlocker::BallotFindLsb: {
        const IR::Value mask =
            ir.CompositeConstruct(ir.Imm32(1), ir.Imm32(0), ir.Imm32(0), ir.Imm32(0));
        ir.Reference(ir.BallotFindLsb(mask));
        break;
    }
    case CrossLaneBlocker::GroupAny:
        ir.Reference(ir.GroupAny(ir.Imm1(true)));
        break;
    }
}

class LowerLaneIdCrossLaneBlockerTest : public ::testing::TestWithParam<CrossLaneBlocker> {};

TEST_P(LowerLaneIdCrossLaneBlockerTest, KeepsNativeLaneIdWhenLiveBlockerSurvives) {
    LaneIdPassFixture fixture;
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.LaneId());
    EmitLiveBlocker(ir, GetParam());

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::GetAttributeU32), 0U);
}

INSTANTIATE_TEST_SUITE_P(
    AllBlockers, LowerLaneIdCrossLaneBlockerTest,
    ::testing::Values(CrossLaneBlocker::WarpId, CrossLaneBlocker::QuadShuffle,
                      CrossLaneBlocker::ReadFirstLane, CrossLaneBlocker::ReadLane,
                      CrossLaneBlocker::WriteLane, CrossLaneBlocker::Ballot,
                      CrossLaneBlocker::BallotFindLsb, CrossLaneBlocker::GroupAny),
    [](const ::testing::TestParamInfo<CrossLaneBlocker>& info) { return BlockerName(info.param); });

class LowerLaneIdDataRingTest : public ::testing::TestWithParam<bool> {};

TEST_P(LowerLaneIdDataRingTest, DeclinesWhenAppendOrConsumeSurvivesWithoutUses) {
    LaneIdPassFixture fixture;
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.LaneId());
    if (GetParam()) {
        static_cast<void>(ir.DataAppend(ir.Imm32(0)));
    } else {
        static_cast<void>(ir.DataConsume(ir.Imm32(0)));
    }

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::GetAttributeU32), 0U);
}

INSTANTIATE_TEST_SUITE_P(AppendAndConsume, LowerLaneIdDataRingTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                             return info.param ? "DataAppend" : "DataConsume";
                         });

struct BoundaryCase {
    const char* name;
    Stage physical_stage;
    u32 subgroup_size;
    std::array<u32, 3> workgroup_size;
};

class LowerLaneIdBoundaryTest : public ::testing::TestWithParam<BoundaryCase> {};

TEST_P(LowerLaneIdBoundaryTest, DeclinesOutsideProvenExecutionShape) {
    const auto& param = GetParam();
    LaneIdPassFixture fixture{param.workgroup_size, param.subgroup_size, param.physical_stage};
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.LaneId());

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::GetAttributeU32), 0U);
}

INSTANTIATE_TEST_SUITE_P(
    StageSubgroupAndWorkgroup, LowerLaneIdBoundaryTest,
    ::testing::Values(BoundaryCase{"NonComputePhysicalStage", Stage::Vertex, 32, {4, 2, 8}},
                      BoundaryCase{"UnknownSubgroupSize", Stage::Compute, 0, {4, 2, 8}},
                      BoundaryCase{"NativeWave64", Stage::Compute, 64, {4, 2, 8}},
                      BoundaryCase{"WiderHostSubgroup", Stage::Compute, 128, {4, 2, 8}},
                      BoundaryCase{"SixtyThreeThreads", Stage::Compute, 32, {63, 1, 1}},
                      BoundaryCase{"SixtyFiveThreads", Stage::Compute, 32, {65, 1, 1}},
                      BoundaryCase{"OneHundredFortyFourThreads", Stage::Compute, 32, {12, 12, 1}},
                      BoundaryCase{"U32ProductOverflow", Stage::Compute, 32, {65536, 65536, 1}},
                      BoundaryCase{"U64ProductWrapsToGuestWave64",
                                   Stage::Compute,
                                   32,
                                   {64U, 107367629U, 2684518405U}}),
    [](const ::testing::TestParamInfo<BoundaryCase>& info) { return info.param.name; });

TEST(LowerLaneIdToLocalInvocationIndexPass, IgnoresDeadCrossLaneInstruction) {
    LaneIdPassFixture fixture;
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.LaneId());
    static_cast<void>(ir.ReadFirstLane(ir.Imm32(1)));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::GetAttributeU32), 3U);
}

TEST(LowerLaneIdToLocalInvocationIndexPass, DoesNothingWithoutLiveLaneId) {
    LaneIdPassFixture fixture;
    IR::IREmitter ir{*fixture.block};
    static_cast<void>(ir.LaneId());

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::GetAttributeU32), 0U);
}

TEST(LowerLaneIdToLocalInvocationIndexPass, AllowsBarrierAndSharedMemoryOperations) {
    LaneIdPassFixture fixture;
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.LaneId());
    ir.WriteShared(32, ir.Imm32(7), ir.Imm32(0));
    ir.Barrier();
    ir.Reference(ir.LoadShared(32, false, ir.Imm32(0)));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::Barrier));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::WriteSharedU32));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::LoadSharedU32));
}

TEST(LowerLaneIdToLocalInvocationIndexPass,
     LowersSupportedCrossLaneOperationsOnlyForUniformStraightLineWave64) {
    LaneIdPassFixture fixture{{64, 1, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    SetStraightLineSyntax(fixture);
    IR::IREmitter ir{*fixture.block};
    const IR::U32 invocation = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    ir.Reference(ir.LaneId());
    ir.Reference(ir.ReadLane(invocation, ir.Imm32(0)));
    ir.Reference(ir.ReadFirstLane(invocation));
    ir.Reference(ir.Ballot(ir.Imm1(true)));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::ReadLane));
    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::ReadFirstLane));
    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::Ballot));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::WriteSharedU32));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::SharedAtomicOr32));
    EXPECT_GE(CountOpcode(*fixture.block, IR::Opcode::Barrier), 7U);
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 64U * sizeof(u32));
}

TEST(LowerLaneIdToLocalInvocationIndexPass, DeclinesUndefinedBallotPredicate) {
    LaneIdPassFixture fixture{{64, 1, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    SetStraightLineSyntax(fixture);
    auto* const undefined_predicate =
        &*fixture.block->PrependNewInst(fixture.block->end(), IR::Opcode::UndefU1);
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.LaneId());
    ir.Reference(ir.Ballot(IR::U1{undefined_predicate}));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::Ballot));
    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::GetAttributeU32), 0U);
}

TEST(LowerLaneIdToLocalInvocationIndexPass, DeclinesIdentityWrappedUndefinedBallotPredicate) {
    LaneIdPassFixture fixture{{64, 1, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    SetStraightLineSyntax(fixture);
    auto* const undefined_predicate =
        &*fixture.block->PrependNewInst(fixture.block->end(), IR::Opcode::UndefU1);
    auto* const identity_predicate = &*fixture.block->PrependNewInst(
        fixture.block->end(), IR::Opcode::Identity, {IR::Value{undefined_predicate}});
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.LaneId());
    ir.Reference(ir.Ballot(IR::U1{identity_predicate}));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::Ballot));
    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::GetAttributeU32), 0U);
}

TEST(LowerLaneIdToLocalInvocationIndexPass, DeclinesPhiWithUndefinedBallotPredicate) {
    LaneIdPassFixture fixture{{8, 8, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    IR::IREmitter entry_ir{*fixture.block};
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U1 condition = entry_ir.IEqual(x, entry_ir.Imm32(0));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);

    auto* const undefined_predicate =
        &*blocks.first_body->PrependNewInst(blocks.first_body->end(), IR::Opcode::UndefU1);
    auto* const phi = &*blocks.merge->PrependNewInst(blocks.merge->begin(), IR::Opcode::Phi);
    phi->SetFlags(IR::Type::U1);
    phi->AddPhiOperand(fixture.block, IR::Value{true});
    phi->AddPhiOperand(blocks.first_body, IR::Value{undefined_predicate});

    IR::IREmitter merge_ir{*blocks.merge};
    merge_ir.Barrier();
    merge_ir.Reference(merge_ir.Ballot(IR::U1{phi}));
    merge_ir.Barrier();

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::Ballot));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
}

TEST(LowerLaneIdToLocalInvocationIndexPass, DeclinesSelectWithUndefinedBallotPredicate) {
    LaneIdPassFixture fixture{{64, 1, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    SetStraightLineSyntax(fixture);
    auto* const undefined_predicate =
        &*fixture.block->PrependNewInst(fixture.block->end(), IR::Opcode::UndefU1);
    IR::IREmitter ir{*fixture.block};
    const IR::U1 condition = ir.IEqual(ir.LaneId(), ir.Imm32(0));
    const IR::U1 predicate{ir.Select(condition, ir.Imm1(true), IR::U1{undefined_predicate})};
    ir.Reference(ir.Ballot(predicate));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::Ballot));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
}

TEST(LowerLaneIdToLocalInvocationIndexPass,
     DeclinesFoldedDirectScalarWordReconstructionBallotOnSubgroup32) {
    LaneIdPassFixture fixture{{64, 1, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    SetStraightLineSyntax(fixture);
    IR::IREmitter ir{*fixture.block};
    const IR::U32 scalar = ir.GetUserData(IR::ScalarReg::S0);
    const IR::U32 shift = ir.BitwiseAnd(ir.LaneId(), ir.Imm32(31));
    const IR::U32 shifted = ir.ShiftRightLogical(scalar, shift);
    const IR::U1 bit = ir.INotEqual(ir.BitwiseAnd(shifted, ir.Imm32(1)), ir.Imm32(0));
    ir.Reference(ir.Ballot(bit));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::Ballot));
    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::SharedAtomicOr32));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
}

TEST(LowerLaneIdToLocalInvocationIndexPass, LowersBallotWithArbitraryDefinedPredicate) {
    LaneIdPassFixture fixture{{64, 1, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    SetStraightLineSyntax(fixture);
    IR::IREmitter ir{*fixture.block};
    const IR::U32 invocation = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U1 predicate = ir.IEqual(invocation, ir.Imm32(37));
    ir.Reference(ir.LaneId());
    ir.Reference(ir.Ballot(predicate));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::Ballot));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::SharedAtomicOr32));
}

TEST(LowerLaneIdToLocalInvocationIndexPass,
     LowersCrossLaneOperationsInBarrierBracketedStructuredMergeBlock) {
    LaneIdPassFixture fixture{{8, 8, 1}};
    fixture.runtime_info.cs_info.shared_memory_size = 1024;
    fixture.profile.max_shared_memory_size = 1280;
    fixture.profile.supports_workgroup_explicit_memory_layout = true;
    IR::IREmitter entry_ir{*fixture.block};
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U1 first_condition = entry_ir.IEqual(x, entry_ir.Imm32(0));
    const IR::U1 second_condition = entry_ir.IEqual(x, entry_ir.Imm32(7));
    const auto blocks =
        SetTwoSelectionBracketedMergeSyntax(fixture, first_condition, second_condition);

    IR::IREmitter merge_ir{*blocks.merge};
    merge_ir.Reference(merge_ir.LoadShared(64, false, merge_ir.Imm32(0)));
    merge_ir.Barrier();
    const IR::U32 lane = merge_ir.LaneId();
    const IR::U32 peer = merge_ir.ReadLane(x, merge_ir.BitwiseXor(lane, merge_ir.Imm32(16)));
    merge_ir.Reference(merge_ir.ReadLane(peer, merge_ir.Imm32(32)));
    merge_ir.Barrier();

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_FALSE(ProgramHasOpcode(fixture.program, IR::Opcode::LaneId));
    EXPECT_FALSE(ProgramHasOpcode(fixture.program, IR::Opcode::ReadLane));
    EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::WriteSharedU32));
    EXPECT_GE(CountOpcode(*blocks.merge, IR::Opcode::Barrier), 6U);
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 256U);
}

class BracketedMergeBarrierTest : public ::testing::TestWithParam<std::pair<bool, bool>> {};

TEST_P(BracketedMergeBarrierTest, DeclinesWhenExistingBarrierDoesNotBracketLaneRegion) {
    const auto [leading_barrier, trailing_barrier] = GetParam();
    LaneIdPassFixture fixture{{8, 8, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    IR::IREmitter entry_ir{*fixture.block};
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U1 condition = entry_ir.IEqual(x, entry_ir.Imm32(0));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);
    IR::IREmitter merge_ir{*blocks.merge};
    if (leading_barrier) {
        merge_ir.Barrier();
    }
    merge_ir.Reference(merge_ir.ReadLane(x, merge_ir.LaneId()));
    if (trailing_barrier) {
        merge_ir.Barrier();
    }

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::ReadLane));
    EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::LaneId));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
}

INSTANTIATE_TEST_SUITE_P(MissingLeadingOrTrailingBarrier, BracketedMergeBarrierTest,
                         ::testing::Values(std::pair{false, true}, std::pair{true, false},
                                           std::pair{false, false}));

TEST(LowerLaneIdToLocalInvocationIndexPass, DeclinesCrossLaneOperationInSelectionBody) {
    LaneIdPassFixture fixture{{8, 8, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    IR::IREmitter entry_ir{*fixture.block};
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U1 condition = entry_ir.IEqual(x, entry_ir.Imm32(0));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);
    IR::IREmitter body_ir{*blocks.first_body};
    body_ir.Barrier();
    body_ir.Reference(body_ir.ReadLane(x, body_ir.LaneId()));
    body_ir.Barrier();

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*blocks.first_body, IR::Opcode::ReadLane));
    EXPECT_TRUE(HasOpcode(*blocks.first_body, IR::Opcode::LaneId));
}

TEST(LowerLaneIdToLocalInvocationIndexPass, DeclinesCrossLaneOperationsSplitAcrossBlocks) {
    LaneIdPassFixture fixture{{8, 8, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    IR::IREmitter entry_ir{*fixture.block};
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U1 condition = entry_ir.IEqual(x, entry_ir.Imm32(0));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);
    IR::IREmitter merge_ir{*blocks.merge};
    merge_ir.Barrier();
    merge_ir.Reference(merge_ir.ReadLane(x, merge_ir.LaneId()));
    merge_ir.Barrier();
    IR::IREmitter later_ir{*blocks.second_body};
    later_ir.Reference(later_ir.ReadFirstLane(x));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::ReadLane));
    EXPECT_TRUE(HasOpcode(*blocks.second_body, IR::Opcode::ReadFirstLane));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
}

TEST(LowerLaneIdToLocalInvocationIndexPass, DeclinesMergeWhoseBodyDoesNotFallThrough) {
    LaneIdPassFixture fixture{{8, 8, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    auto* const body = fixture.pools.block_pool.Create(fixture.pools.inst_pool);
    auto* const merge = fixture.pools.block_pool.Create(fixture.pools.inst_pool);
    fixture.block->AddBranch(body);
    fixture.block->AddBranch(merge);
    fixture.program.blocks = {fixture.block, body, merge};
    IR::IREmitter entry_ir{*fixture.block};
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U1 condition = entry_ir.IEqual(x, entry_ir.Imm32(0));
    AppendBlockNode(fixture.program, fixture.block);
    AppendIfNode(fixture.program, condition, body, merge);
    AppendBlockNode(fixture.program, body);
    AppendEndIfNode(fixture.program, merge);
    AppendBlockNode(fixture.program, merge);
    fixture.program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    IR::IREmitter merge_ir{*merge};
    merge_ir.Barrier();
    merge_ir.Reference(merge_ir.ReadLane(x, merge_ir.LaneId()));
    merge_ir.Barrier();

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*merge, IR::Opcode::ReadLane));
    EXPECT_TRUE(HasOpcode(*merge, IR::Opcode::LaneId));
}

enum class StructuredControlBlocker { EarlyReturn, Break, Loop };

class BracketedMergeStructuredControlTest
    : public ::testing::TestWithParam<StructuredControlBlocker> {};

TEST_P(BracketedMergeStructuredControlTest, DeclinesNonFallthroughOrLoopingControlFlow) {
    LaneIdPassFixture fixture{{8, 8, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    IR::IREmitter entry_ir{*fixture.block};
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U1 condition = entry_ir.IEqual(x, entry_ir.Imm32(0));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);
    IR::IREmitter merge_ir{*blocks.merge};
    merge_ir.Barrier();
    merge_ir.Reference(merge_ir.ReadLane(x, merge_ir.LaneId()));
    merge_ir.Barrier();

    IR::AbstractSyntaxNode blocker{};
    switch (GetParam()) {
    case StructuredControlBlocker::EarlyReturn:
        blocker.type = IR::AbstractSyntaxNode::Type::Return;
        break;
    case StructuredControlBlocker::Break:
        blocker.type = IR::AbstractSyntaxNode::Type::Break;
        blocker.data.break_node.cond = condition;
        blocker.data.break_node.merge = blocks.merge;
        blocker.data.break_node.skip = blocks.first_body;
        break;
    case StructuredControlBlocker::Loop:
        blocker.type = IR::AbstractSyntaxNode::Type::Loop;
        blocker.data.loop.body = blocks.first_body;
        blocker.data.loop.continue_block = blocks.first_body;
        blocker.data.loop.merge = blocks.merge;
        break;
    }
    fixture.program.syntax_list.insert(fixture.program.syntax_list.begin() + 3, blocker);

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::ReadLane));
    EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::LaneId));
}

INSTANTIATE_TEST_SUITE_P(AllUnsupportedStructuredControl, BracketedMergeStructuredControlTest,
                         ::testing::Values(StructuredControlBlocker::EarlyReturn,
                                           StructuredControlBlocker::Break,
                                           StructuredControlBlocker::Loop));

TEST(LowerLaneIdToLocalInvocationIndexPass,
     DeclinesBracketedMergeWithNonU32LdsWithoutExplicitLayout) {
    LaneIdPassFixture fixture{{8, 8, 1}};
    fixture.runtime_info.cs_info.shared_memory_size = 1024;
    fixture.profile.max_shared_memory_size = 32 * 1024;
    fixture.profile.supports_workgroup_explicit_memory_layout = false;
    IR::IREmitter entry_ir{*fixture.block};
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U1 condition = entry_ir.IEqual(x, entry_ir.Imm32(0));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);
    IR::IREmitter merge_ir{*blocks.merge};
    merge_ir.Reference(merge_ir.LoadShared(64, false, merge_ir.Imm32(0)));
    merge_ir.Barrier();
    merge_ir.Reference(merge_ir.ReadLane(x, merge_ir.LaneId()));
    merge_ir.Barrier();

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::ReadLane));
    EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::LaneId));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
}

TEST(LowerLaneIdToLocalInvocationIndexPass, DeclinesBracketedMergeWhenScratchExceedsHostLds) {
    LaneIdPassFixture fixture{{8, 8, 1}};
    fixture.runtime_info.cs_info.shared_memory_size = 1024;
    fixture.profile.max_shared_memory_size = 1279;
    IR::IREmitter entry_ir{*fixture.block};
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U1 condition = entry_ir.IEqual(x, entry_ir.Imm32(0));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);
    IR::IREmitter merge_ir{*blocks.merge};
    merge_ir.Barrier();
    merge_ir.Reference(merge_ir.ReadLane(x, merge_ir.LaneId()));
    merge_ir.Barrier();

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::ReadLane));
    EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::LaneId));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
}

TEST(LowerLaneIdToLocalInvocationIndexPass, DeclinesCrossLaneLoweringWhenScratchExceedsHostLds) {
    LaneIdPassFixture fixture{{64, 1, 1}};
    fixture.profile.max_shared_memory_size = 255;
    SetStraightLineSyntax(fixture);
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.LaneId());
    ir.Reference(ir.ReadFirstLane(ir.Imm32(1)));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::ReadFirstLane));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
}

TEST(LowerLaneIdToLocalInvocationIndexPass, DeclinesCrossLaneLoweringWithGuestExecState) {
    LaneIdPassFixture fixture{{64, 1, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    SetStraightLineSyntax(fixture);
    IR::IREmitter ir{*fixture.block};
    ir.SetExec(ir.Imm1(true));
    ir.Reference(ir.LaneId());
    ir.Reference(ir.ReadFirstLane(ir.Imm32(1)));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::ReadFirstLane));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
}

TEST(LowerLaneIdToLocalInvocationIndexPass, AppendsAlignedScratchAfterOddSizedGuestSharedMemory) {
    LaneIdPassFixture fixture{{64, 1, 1}};
    fixture.runtime_info.cs_info.shared_memory_size = 3;
    fixture.profile.max_shared_memory_size = 260;
    SetStraightLineSyntax(fixture);
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.ReadLane(ir.Imm32(7), ir.Imm32(63)));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::ReadLane));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 257U);
}

TEST(CollectShaderInfoPass, DirectBallotRequestsBallotCapability) {
    LaneIdPassFixture fixture{{64, 1, 1}};
    fixture.program.post_order_blocks.push_back(fixture.block);
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.Ballot(ir.Imm1(true)));

    Optimization::CollectShaderInfoPass(fixture.program, fixture.profile);

    EXPECT_TRUE(fixture.info.uses_group_ballot);
}

bool HasBuiltIn(const std::vector<u32>& spirv, spv::BuiltIn builtin) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (words == 0 || offset + words > spirv.size()) {
            return false;
        }
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpDecorate && words >= 4 &&
            spirv[offset + 2] == static_cast<u32>(spv::DecorationBuiltIn) &&
            spirv[offset + 3] == static_cast<u32>(builtin)) {
            return true;
        }
        offset += words;
    }
    return false;
}

bool HasSpirvOpcode(const std::vector<u32>& spirv, spv::Op opcode) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (words == 0 || offset + words > spirv.size()) {
            return false;
        }
        if (static_cast<spv::Op>(word & 0xffff) == opcode) {
            return true;
        }
        offset += words;
    }
    return false;
}

#ifdef SHADPS4_SPIRV_VAL_PATH
int RunSpirvValidator(const std::filesystem::path& module) {
#ifdef _WIN32
    const std::string module_path = module.string();
    return static_cast<int>(_spawnl(_P_WAIT, SHADPS4_SPIRV_VAL_PATH, SHADPS4_SPIRV_VAL_PATH,
                                    "--target-env", "vulkan1.3", module_path.c_str(), nullptr));
#else
    const std::string command = std::string{"\""} + SHADPS4_SPIRV_VAL_PATH +
                                "\" --target-env vulkan1.3 \"" + module.string() + "\"";
    return std::system(command.c_str());
#endif
}
#endif

TEST(LowerLaneIdToLocalInvocationIndexPass, EmitsValidSpirvWithoutSubgroupLaneBuiltin) {
    LaneIdPassFixture fixture;
    fixture.program.syntax_list.emplace_back();
    fixture.program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    fixture.program.syntax_list.back().data.block = fixture.block;
    fixture.program.syntax_list.emplace_back();
    fixture.program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    fixture.program.post_order_blocks = IR::PostOrder(fixture.program.syntax_list.front());

    IR::IREmitter ir{*fixture.block};
    ir.Prologue();
    ir.Reference(ir.LaneId());
    ir.Epilogue();

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);
    Optimization::CollectShaderInfoPass(fixture.program, fixture.profile);
    Backend::Bindings bindings{};
    const auto spirv =
        Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.runtime_info, fixture.program, bindings);

    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInLocalInvocationId));
    EXPECT_FALSE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));

#ifdef SHADPS4_SPIRV_VAL_PATH
    const auto output =
        std::filesystem::temp_directory_path() / "shadps4-lane-id-lowering-test.spv";
    {
        std::ofstream file{output, std::ios::binary | std::ios::trunc};
        ASSERT_TRUE(file.good());
        file.write(reinterpret_cast<const char*>(spirv.data()),
                   static_cast<std::streamsize>(spirv.size() * sizeof(u32)));
    }
    EXPECT_EQ(RunSpirvValidator(output), 0);
    std::error_code remove_error;
    std::filesystem::remove(output, remove_error);
#endif
}

TEST(LowerLaneIdToLocalInvocationIndexPass,
     DeclinesSsaMaterializedMixedScalarWordBallotOnSubgroup32) {
    LaneIdPassFixture fixture{{8, 8, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    IR::IREmitter entry_ir{*fixture.block};
    entry_ir.Prologue();
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U1 condition = entry_ir.ConditionRef(entry_ir.IEqual(x, entry_ir.Imm32(0)));
    entry_ir.SetScalarReg(IR::ScalarReg::S2, entry_ir.Imm32(0xA5A5A5A5));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);

    IR::IREmitter body_ir{*blocks.first_body};
    body_ir.SetThreadBitScalarRegPair(IR::ScalarReg::S1, body_ir.Imm1(true));
    IR::IREmitter merge_ir{*blocks.merge};
    merge_ir.Barrier();
    merge_ir.Reference(merge_ir.GetScalarReg(IR::ScalarReg::S2));
    merge_ir.Barrier();
    IR::IREmitter{*blocks.exit}.Epilogue();
    fixture.program.post_order_blocks = IR::PostOrder(fixture.program.syntax_list.front());

    Optimization::SsaRewritePass(fixture.program);
    ASSERT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::Ballot));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);
    EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::Ballot));
    EXPECT_FALSE(HasOpcode(*blocks.merge, IR::Opcode::SharedAtomicOr32));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
}

} // namespace
