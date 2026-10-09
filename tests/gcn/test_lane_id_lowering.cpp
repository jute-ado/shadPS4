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

TEST(LowerLaneIdToLocalInvocationIndexPass,
     LowersCrossLaneOperationsInUniformEntryBlockBeforeStructuredControl) {
    LaneIdPassFixture fixture{{8, 8, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    IR::IREmitter entry_ir{*fixture.block};
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    entry_ir.SetExec(entry_ir.Imm1(true));
    const IR::U32 lane = entry_ir.LaneId();
    const IR::U32 peer =
        entry_ir.ReadLane(x, entry_ir.BitwiseXor(lane, entry_ir.Imm32(16)));
    entry_ir.SetExec(entry_ir.Imm1(true));
    entry_ir.Reference(entry_ir.ReadLane(peer, entry_ir.Imm32(32)));
    entry_ir.SetExec(entry_ir.Imm1(false));
    const IR::U1 condition = entry_ir.IEqual(x, entry_ir.Imm32(0));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);
    IR::IREmitter later_ir{*blocks.first_body};
    later_ir.SetExec(later_ir.Imm1(false));
    later_ir.Reference(later_ir.ReadFirstLane(x));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::ReadLane));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::WriteSharedU32));
    EXPECT_GE(CountOpcode(*fixture.block, IR::Opcode::Barrier), 4U);
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 64U * sizeof(u32));
    EXPECT_TRUE(HasOpcode(*blocks.first_body, IR::Opcode::ReadFirstLane));
    EXPECT_TRUE(HasOpcode(*blocks.first_body, IR::Opcode::SetExec));
}

TEST(LowerLaneIdToLocalInvocationIndexPass,
     DeclinesEntryRegionLoweringWhenLaneIdRemainsLiveInLaterControlFlow) {
    LaneIdPassFixture fixture{{8, 8, 1}};
    fixture.profile.max_shared_memory_size = 32 * 1024;
    IR::IREmitter entry_ir{*fixture.block};
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U32 entry_lane = entry_ir.LaneId();
    entry_ir.Reference(entry_ir.ReadLane(x, entry_lane));
    const IR::U1 condition = entry_ir.IEqual(x, entry_ir.Imm32(0));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);
    IR::IREmitter later_ir{*blocks.first_body};
    later_ir.Reference(later_ir.LaneId());
    later_ir.Reference(later_ir.ReadLane(x, entry_lane));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::ReadLane));
    EXPECT_TRUE(HasOpcode(*blocks.first_body, IR::Opcode::LaneId));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
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

TEST(LowerLaneIdToLocalInvocationIndexPass,
     ConfinedEntryWaveBridgeDoesNotRewriteLaterNativeLaneOperations) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    fixture.profile.max_shared_memory_size = 32U * 1024U;
    IR::IREmitter entry_ir{*fixture.block};
    entry_ir.SetExec(entry_ir.Imm1(true));
    const auto x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0U);
    const auto y = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1U);
    const IR::U32 value{entry_ir.IAdd(x, entry_ir.IMul(y, entry_ir.Imm32(8U)))};
    const auto entry_lane = entry_ir.LaneId();
    const auto peer = entry_ir.ReadLane(value, entry_ir.BitwiseXor(entry_lane, entry_ir.Imm32(16U)));
    const auto lower_half = entry_ir.ReadLane(peer, entry_ir.Imm32(0U));
    const auto upper_half = entry_ir.ReadLane(peer, entry_ir.Imm32(32U));
    const IR::U32 entry_result{entry_ir.IAdd(lower_half, upper_half)};
    const auto condition = entry_ir.IEqual(x, entry_ir.Imm32(0U));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);
    IR::IREmitter later_ir{*blocks.first_body};
    const auto later_lane = later_ir.LaneId();
    const auto later_target = later_ir.BitwiseXor(later_lane, later_ir.Imm32(16U));
    const auto later_peer = later_ir.ReadLane(value, later_target);
    later_ir.Reference(later_ir.IAdd(later_peer, entry_result));

    // Every direct use of the entry-local LaneId remains in the uniform entry region. Its
    // completed reduction may escape; an unlowered native lane selector must not escape.
    for (const auto& use : entry_lane.InstRecursive()->Uses()) {
        ASSERT_EQ(use.user->GetParent(), fixture.block);
    }
    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::ReadLane));
    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::WriteSharedU32), 3U);
    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::Barrier), 6U);
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 64U * sizeof(u32));
    EXPECT_EQ(later_lane.InstRecursive()->GetOpcode(), IR::Opcode::LaneId);
    EXPECT_EQ(later_peer.InstRecursive()->GetOpcode(), IR::Opcode::ReadLane);
    EXPECT_EQ(later_peer.InstRecursive()->Arg(1), later_target);
    for (auto* const block : fixture.program.blocks) {
        if (block != fixture.block) {
            EXPECT_EQ(CountOpcode(*block, IR::Opcode::Barrier), 0U);
            EXPECT_EQ(CountOpcode(*block, IR::Opcode::WriteSharedU32), 0U);
        }
    }
}

TEST(LowerLaneIdToLocalInvocationIndexPass,
     EntryLaneIdEscapingIntoLaterNativeSelectorRemainsConservative) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    fixture.profile.max_shared_memory_size = 32U * 1024U;
    IR::IREmitter entry_ir{*fixture.block};
    entry_ir.SetExec(entry_ir.Imm1(true));
    const auto x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0U);
    const auto entry_lane = entry_ir.LaneId();
    entry_ir.Reference(entry_ir.ReadLane(x, entry_lane));
    const auto condition = entry_ir.IEqual(x, entry_ir.Imm32(0U));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);
    IR::IREmitter later_ir{*blocks.first_body};
    later_ir.Reference(later_ir.ReadLane(x, entry_lane));
    later_ir.Reference(later_ir.LaneId());

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::ReadLane));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
    EXPECT_FALSE(ProgramHasOpcode(fixture.program, IR::Opcode::Barrier));
}

TEST(LowerLaneIdToLocalInvocationIndexPass,
     EntryLaneIdIndirectlyEscapingIntoLaterNativeSelectorRemainsConservative) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    fixture.profile.max_shared_memory_size = 32U * 1024U;
    IR::IREmitter entry_ir{*fixture.block};
    entry_ir.SetExec(entry_ir.Imm1(true));
    const auto x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0U);
    const auto entry_lane = entry_ir.LaneId();
    const IR::U32 entry_target{entry_ir.BitwiseXor(entry_lane, entry_ir.Imm32(16U))};
    entry_ir.Reference(entry_ir.ReadLane(x, entry_target));
    const auto condition = entry_ir.IEqual(x, entry_ir.Imm32(0U));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);
    IR::IREmitter later_ir{*blocks.first_body};
    later_ir.Reference(later_ir.ReadLane(x, entry_target));
    later_ir.Reference(later_ir.LaneId());

    // A direct-user-only check would pass: the lane's XOR lives in entry, while its result
    // escapes into a later native subgroup selector. Admission must follow the complete use path.
    for (const auto& use : entry_lane.InstRecursive()->Uses()) {
        ASSERT_EQ(use.user->GetParent(), fixture.block);
    }
    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::ReadLane));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
    EXPECT_FALSE(ProgramHasOpcode(fixture.program, IR::Opcode::Barrier));
}

TEST(LowerLaneIdToLocalInvocationIndexPass,
     EntryLaneIdEscapeIsRejectedWithoutASeparateLaterLaneIdInstruction) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    fixture.profile.max_shared_memory_size = 32U * 1024U;
    IR::IREmitter entry_ir{*fixture.block};
    entry_ir.SetExec(entry_ir.Imm1(true));
    const auto x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0U);
    const auto entry_lane = entry_ir.LaneId();
    const IR::U32 entry_target{entry_ir.BitwiseXor(entry_lane, entry_ir.Imm32(16U))};
    entry_ir.Reference(entry_ir.ReadLane(x, entry_target));
    const auto condition = entry_ir.IEqual(x, entry_ir.Imm32(0U));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);
    IR::IREmitter later_ir{*blocks.first_body};
    later_ir.Reference(later_ir.ReadLane(x, entry_target));

    // Looking only for LaneId instructions outside entry misses an entry-defined lane index
    // reused by a later native subgroup operation.
    ASSERT_FALSE(HasOpcode(*blocks.first_body, IR::Opcode::LaneId));
    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::ReadLane));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
    EXPECT_FALSE(ProgramHasOpcode(fixture.program, IR::Opcode::Barrier));
}

TEST(LowerLaneIdToLocalInvocationIndexPass,
     EntryLaneIdEscapingThroughASyntaxConditionRemainsConservative) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    fixture.profile.max_shared_memory_size = 32U * 1024U;
    IR::IREmitter entry_ir{*fixture.block};
    entry_ir.SetExec(entry_ir.Imm1(true));
    const auto x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0U);
    const auto entry_lane = entry_ir.LaneId();
    entry_ir.Reference(entry_ir.ReadLane(x, entry_lane));
    const auto condition = entry_ir.IEqual(entry_lane, entry_ir.Imm32(0U));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);
    IR::IREmitter later_ir{*blocks.first_body};
    later_ir.Reference(later_ir.ReadLane(x, later_ir.Imm32(0U)));

    // Syntax conditions are external graph references rather than ordinary instruction uses.
    // Changing this condition to guest lane 0 changes participation in later native operations.
    ASSERT_FALSE(condition.InstRecursive()->HasUses());
    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::ReadLane));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
    EXPECT_FALSE(ProgramHasOpcode(fixture.program, IR::Opcode::Barrier));
}

TEST(LowerLaneIdToLocalInvocationIndexPass,
     EntryLaneIdEscapingThroughExecAfterLastWaveOperationRemainsConservative) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    fixture.profile.max_shared_memory_size = 32U * 1024U;
    IR::IREmitter entry_ir{*fixture.block};
    entry_ir.SetExec(entry_ir.Imm1(true));
    const auto x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0U);
    const auto entry_lane = entry_ir.LaneId();
    entry_ir.Reference(entry_ir.ReadLane(x, entry_lane));
    entry_ir.SetExec(entry_ir.IEqual(entry_lane, entry_ir.Imm32(0U)));
    const auto condition = entry_ir.IEqual(x, entry_ir.Imm32(0U));
    const auto blocks = SetTwoSelectionBracketedMergeSyntax(fixture, condition, condition);
    IR::IREmitter later_ir{*blocks.first_body};
    later_ir.Reference(later_ir.ReadFirstLane(x));

    // SetExec is retained specially by DCE but is not classified by MayHaveSideEffects().
    // The last entry read has full EXEC; the lane-dependent update afterward still escapes.
    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::LaneId));
    EXPECT_TRUE(HasOpcode(*fixture.block, IR::Opcode::ReadLane));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
    EXPECT_FALSE(ProgramHasOpcode(fixture.program, IR::Opcode::Barrier));
}

enum class FirstBodyCondition { FullWaveReadResult, LocalInvocation, SharedMemory };
enum class FirstBodyExec { ExplicitFull, InheritedFull, Partial, Unknown };

IR::Block* BuildFirstUniformBodyProbe(LaneIdPassFixture& fixture,
                                    FirstBodyCondition condition_kind,
                                    FirstBodyExec body_exec) {
    fixture.profile.max_shared_memory_size = 32U * 1024U;
    fixture.runtime_info.cs_info.shared_memory_size = 1024U;
    IR::IREmitter entry_ir{*fixture.block};
    entry_ir.SetExec(entry_ir.Imm1(true));
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0U);
    const IR::U32 y = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1U);
    const IR::U32 value = entry_ir.IAdd(x, entry_ir.IMul(y, entry_ir.Imm32(8U)));
    const auto low = entry_ir.ReadLane(value, entry_ir.Imm32(0U));
    const auto high = entry_ir.ReadLane(value, entry_ir.Imm32(32U));
    const IR::U32 combined{entry_ir.BitwiseOr(low, high)};
    entry_ir.Reference(combined);
    IR::U1 condition = entry_ir.INotEqual(combined, entry_ir.Imm32(0U));
    if (condition_kind == FirstBodyCondition::LocalInvocation) {
        condition = entry_ir.IEqual(value, entry_ir.Imm32(0U));
    } else if (condition_kind == FirstBodyCondition::SharedMemory) {
        condition = entry_ir.INotEqual(
            IR::U32{entry_ir.LoadShared(32, false, entry_ir.Imm32(0U))}, entry_ir.Imm32(0U));
    }
    condition = entry_ir.ConditionRef(condition);
    auto* const body = fixture.pools.block_pool.Create(fixture.pools.inst_pool);
    auto* const exit = fixture.pools.block_pool.Create(fixture.pools.inst_pool);
    fixture.block->AddBranch(body);
    fixture.block->AddBranch(exit);
    body->AddBranch(exit);
    fixture.program.blocks = {fixture.block, body, exit};
    AppendBlockNode(fixture.program, fixture.block);
    AppendIfNode(fixture.program, condition, body, exit);
    AppendBlockNode(fixture.program, body);
    AppendEndIfNode(fixture.program, exit);
    AppendBlockNode(fixture.program, exit);
    fixture.program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;

    IR::IREmitter body_ir{*body};
    if (body_exec == FirstBodyExec::ExplicitFull) {
        body_ir.SetExec(body_ir.Imm1(true));
    } else if (body_exec == FirstBodyExec::Partial) {
        body_ir.SetExec(body_ir.Imm1(false));
    } else if (body_exec == FirstBodyExec::Unknown) {
        body_ir.SetExec(body_ir.IEqual(value, body_ir.Imm32(0U)));
    }
    const auto lane = body_ir.LaneId();
    const auto peer = body_ir.ReadLane(value, body_ir.BitwiseXor(lane, body_ir.Imm32(16U)));
    const auto body_low = body_ir.ReadLane(peer, body_ir.Imm32(0U));
    const auto body_high = body_ir.ReadLane(peer, body_ir.Imm32(32U));
    body_ir.Reference(body_ir.BitwiseOr(body_low, body_high));
    return body;
}

TEST(UniformWaveBodyAdmission, FixedEntryReadResultsProveTheFirstBodyUniform) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    auto* const body = BuildFirstUniformBodyProbe(
        fixture, FirstBodyCondition::FullWaveReadResult, FirstBodyExec::ExplicitFull);

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::Barrier), 4U);
    EXPECT_EQ(CountOpcode(*body, IR::Opcode::Barrier), 6U);
    EXPECT_FALSE(HasOpcode(*body, IR::Opcode::ReadLane));
    EXPECT_FALSE(HasOpcode(*body, IR::Opcode::LaneId));
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 256U);
}

TEST(UniformWaveBodyAdmission, FirstBodyInheritsFullExecWithoutARedundantSet) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    auto* const body = BuildFirstUniformBodyProbe(
        fixture, FirstBodyCondition::FullWaveReadResult, FirstBodyExec::InheritedFull);

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::Barrier), 4U);
    EXPECT_EQ(CountOpcode(*body, IR::Opcode::Barrier), 6U);
    EXPECT_FALSE(HasOpcode(*body, IR::Opcode::ReadLane));
    EXPECT_FALSE(HasOpcode(*body, IR::Opcode::LaneId));
}

TEST(UniformWaveBodyAdmission, DivergentOrMemoryDerivedConditionsDoNotProveTheBodyUniform) {
    for (const auto condition : {FirstBodyCondition::LocalInvocation, FirstBodyCondition::SharedMemory}) {
        LaneIdPassFixture fixture{{8U, 8U, 1U}};
        auto* const body =
            BuildFirstUniformBodyProbe(fixture, condition, FirstBodyExec::ExplicitFull);

        Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                            fixture.profile);

        EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::Barrier), 4U);
        EXPECT_EQ(CountOpcode(*body, IR::Opcode::Barrier), 0U);
        EXPECT_TRUE(HasOpcode(*body, IR::Opcode::ReadLane));
        EXPECT_TRUE(HasOpcode(*body, IR::Opcode::LaneId));
    }
}

TEST(UniformWaveBodyAdmission, PartialOrUnknownExecKeepsTheBodyNative) {
    for (const auto exec : {FirstBodyExec::Partial, FirstBodyExec::Unknown}) {
        LaneIdPassFixture fixture{{8U, 8U, 1U}};
        auto* const body =
            BuildFirstUniformBodyProbe(fixture, FirstBodyCondition::FullWaveReadResult, exec);

        Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                            fixture.profile);

        EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::Barrier), 4U);
        EXPECT_EQ(CountOpcode(*body, IR::Opcode::Barrier), 0U);
        EXPECT_TRUE(HasOpcode(*body, IR::Opcode::ReadLane));
        EXPECT_TRUE(HasOpcode(*body, IR::Opcode::LaneId));
    }
}

TEST(UniformWaveBodyAdmission, ASelectedReadWithAVaryingLaneSelectorIsNotAUniformSeed) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    auto* const body = BuildFirstUniformBodyProbe(
        fixture, FirstBodyCondition::FullWaveReadResult, FirstBodyExec::ExplicitFull);
    auto* const read = FindOpcode(*fixture.block, IR::Opcode::ReadLane);
    auto* const local_id = FindOpcode(*fixture.block, IR::Opcode::GetAttributeU32);
    ASSERT_NE(read, nullptr);
    ASSERT_NE(local_id, nullptr);
    read->SetArg(1, IR::Value{local_id});

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::Barrier), 4U);
    EXPECT_EQ(CountOpcode(*body, IR::Opcode::Barrier), 0U);
    EXPECT_TRUE(HasOpcode(*body, IR::Opcode::ReadLane));
}

TEST(UniformWaveBodyAdmission, AChangedOrMultiplyEnteredBodyIsNotAdmitted) {
    for (const bool extra_predecessor : {false, true}) {
        LaneIdPassFixture fixture{{8U, 8U, 1U}};
        auto* const body = BuildFirstUniformBodyProbe(
            fixture, FirstBodyCondition::FullWaveReadResult, FirstBodyExec::ExplicitFull);
        auto* const exit = fixture.program.blocks.back();
        if (extra_predecessor) {
            exit->AddBranch(body);
        } else {
            fixture.program.syntax_list[1].data.if_node.body = exit;
        }

        Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                            fixture.profile);

        EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::Barrier), 4U);
        EXPECT_EQ(CountOpcode(*body, IR::Opcode::Barrier), 0U);
        EXPECT_TRUE(HasOpcode(*body, IR::Opcode::ReadLane));
    }
}

TEST(UniformWaveBodyAdmission, APartialFinalEntryMaskCannotBeInheritedAsFull) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    auto* const body = BuildFirstUniformBodyProbe(
        fixture, FirstBodyCondition::FullWaveReadResult, FirstBodyExec::InheritedFull);
    IR::IREmitter entry_ir{*fixture.block};
    entry_ir.SetExec(entry_ir.Imm1(false));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::Barrier), 4U);
    EXPECT_EQ(CountOpcode(*body, IR::Opcode::Barrier), 0U);
    EXPECT_TRUE(HasOpcode(*body, IR::Opcode::ReadLane));
}

TEST(UniformWaveBodyAdmission, ABodyLaneSelectorCannotEscapeIntoLaterNativeOperations) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    auto* const body = BuildFirstUniformBodyProbe(
        fixture, FirstBodyCondition::FullWaveReadResult, FirstBodyExec::ExplicitFull);
    auto* const lane = FindOpcode(*body, IR::Opcode::LaneId);
    ASSERT_NE(lane, nullptr);
    IR::IREmitter body_ir{*body};
    const IR::U32 selector{body_ir.BitwiseXor(IR::U32{lane}, body_ir.Imm32(16U))};
    auto* const exit = fixture.program.blocks.back();
    auto* const merged_selector =
        &*exit->PrependNewInst(exit->begin(), IR::Opcode::Phi);
    merged_selector->SetFlags(IR::Type::U32);
    merged_selector->AddPhiOperand(body, selector);
    merged_selector->AddPhiOperand(fixture.block, IR::Value{0U});
    IR::IREmitter exit_ir{*exit};
    exit_ir.Reference(exit_ir.ReadLane(exit_ir.Imm32(1U), IR::U32{merged_selector}));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::Barrier), 4U);
    EXPECT_EQ(CountOpcode(*body, IR::Opcode::Barrier), 0U);
    EXPECT_TRUE(HasOpcode(*body, IR::Opcode::ReadLane));
    EXPECT_TRUE(HasOpcode(*body, IR::Opcode::LaneId));
}

TEST(UniformWaveBodyAdmission, UndefinedWaveReadSourcesDoNotProveUniformControl) {
    for (const bool first_lane : {false, true}) {
        for (const bool identity_wrapped : {false, true}) {
            SCOPED_TRACE(first_lane);
            SCOPED_TRACE(identity_wrapped);
            LaneIdPassFixture fixture{{8U, 8U, 1U}};
            auto* const body = BuildFirstUniformBodyProbe(
                fixture, FirstBodyCondition::FullWaveReadResult, FirstBodyExec::ExplicitFull);
            std::vector<IR::Inst*> reads;
            for (IR::Inst& inst : *fixture.block) {
                if (inst.GetOpcode() == IR::Opcode::ReadLane) {
                    reads.push_back(&inst);
                }
            }
            ASSERT_EQ(reads.size(), 2U);
            const auto point = IR::Block::InstructionList::s_iterator_to(*reads.front());
            auto* const undefined =
                &*fixture.block->PrependNewInst(point, IR::Opcode::UndefU32);
            IR::Value source{undefined};
            if (identity_wrapped) {
                source = IR::Value{&*fixture.block->PrependNewInst(point, IR::Opcode::Identity,
                                                                {source})};
            }
            for (auto* const read : reads) {
                read->SetArg(0, source);
                if (first_lane) {
                    read->SetArg(1, IR::Value{});
                    read->ReplaceOpcode(IR::Opcode::ReadFirstLane);
                }
            }

            Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program,
                                                                fixture.runtime_info,
                                                                fixture.profile);

            EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::Barrier), 4U);
            EXPECT_EQ(CountOpcode(*body, IR::Opcode::Barrier), 0U);
            EXPECT_TRUE(HasOpcode(*body, IR::Opcode::ReadLane));
            EXPECT_TRUE(HasOpcode(*body, IR::Opcode::LaneId));
        }
    }
}

struct NestedWaveMergeProbe {
    IR::Block* header;
    IR::Block* first_body;
    IR::Block* nested_body;
    IR::Block* nested_merge;
    IR::Block* second_body;
    IR::Block* second_merge;
    IR::Block* candidate;
    IR::Block* exit;
};

IR::U32 MergeU32(IR::Block& block, IR::Block* first, const IR::U32& first_value,
                IR::Block* second, const IR::U32& second_value) {
    auto* const phi = &*block.PrependNewInst(block.begin(), IR::Opcode::Phi);
    phi->SetFlags(IR::Type::U32);
    phi->AddPhiOperand(first, first_value);
    phi->AddPhiOperand(second, second_value);
    return IR::U32{phi};
}

NestedWaveMergeProbe BuildNestedWaveMergeProbe(LaneIdPassFixture& fixture) {
    fixture.profile.max_shared_memory_size = 32U * 1024U;
    fixture.runtime_info.cs_info.shared_memory_size = 1024U;
    const auto create = [&] { return fixture.pools.block_pool.Create(fixture.pools.inst_pool); };
    NestedWaveMergeProbe blocks{create(), create(), create(), create(), create(), create(), create(),
                               create()};
    fixture.program.blocks = {fixture.block, blocks.header, blocks.first_body, blocks.nested_body,
                              blocks.nested_merge, blocks.second_body, blocks.second_merge,
                              blocks.candidate, blocks.exit};
    fixture.block->AddBranch(blocks.header);
    fixture.block->AddBranch(blocks.exit);
    blocks.header->AddBranch(blocks.first_body);
    blocks.header->AddBranch(blocks.candidate);
    blocks.first_body->AddBranch(blocks.nested_body);
    blocks.first_body->AddBranch(blocks.nested_merge);
    blocks.nested_body->AddBranch(blocks.nested_merge);
    blocks.nested_merge->AddBranch(blocks.second_body);
    blocks.nested_merge->AddBranch(blocks.second_merge);
    blocks.second_body->AddBranch(blocks.second_merge);
    blocks.second_merge->AddBranch(blocks.candidate);
    blocks.candidate->AddBranch(blocks.exit);

    IR::IREmitter entry_ir{*fixture.block};
    entry_ir.SetExec(entry_ir.Imm1(true));
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0U);
    const IR::U32 y = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1U);
    const IR::U32 value = entry_ir.IAdd(x, entry_ir.IMul(y, entry_ir.Imm32(8U)));
    const auto low = entry_ir.ReadLane(value, entry_ir.Imm32(0U));
    const auto high = entry_ir.ReadLane(value, entry_ir.Imm32(32U));
    const IR::U1 outer_condition = entry_ir.ConditionRef(
        entry_ir.INotEqual(entry_ir.BitwiseOr(low, high), entry_ir.Imm32(0U)));
    IR::IREmitter header_ir{*blocks.header};
    header_ir.SetExec(header_ir.Imm1(true));
    header_ir.Reference(header_ir.ReadLane(value, header_ir.Imm32(0U)));
    header_ir.Reference(header_ir.ReadLane(value, header_ir.Imm32(32U)));
    const IR::U1 inner_condition = header_ir.IEqual(
        header_ir.BitwiseAnd(value, header_ir.Imm32(1U)), header_ir.Imm32(0U));
    header_ir.SetExec(inner_condition);
    const auto inner_ref = header_ir.ConditionRef(inner_condition);
    IR::IREmitter first_ir{*blocks.first_body};
    first_ir.Reference(first_ir.LaneId());
    const auto nested_condition = first_ir.ConditionRef(
        first_ir.IEqual(first_ir.BitwiseAnd(value, first_ir.Imm32(2U)), first_ir.Imm32(0U)));
    IR::IREmitter nested_ir{*blocks.nested_body};
    const IR::U32 first_payload = nested_ir.IAdd(value, nested_ir.Imm32(100U));
    const auto first_merge = MergeU32(*blocks.nested_merge, blocks.nested_body, first_payload,
                                     blocks.first_body, value);
    IR::IREmitter merge_ir{*blocks.nested_merge};
    const auto second_condition = merge_ir.ConditionRef(
        merge_ir.IEqual(merge_ir.BitwiseAnd(value, merge_ir.Imm32(4U)), merge_ir.Imm32(0U)));
    IR::IREmitter second_ir{*blocks.second_body};
    const IR::U32 second_payload = second_ir.IAdd(first_merge, second_ir.Imm32(200U));
    const auto second_merge = MergeU32(*blocks.second_merge, blocks.second_body, second_payload,
                                      blocks.nested_merge, first_merge);
    const auto payload = MergeU32(*blocks.candidate, blocks.second_merge, second_merge,
                                 blocks.header, value);
    IR::IREmitter candidate_ir{*blocks.candidate};
    candidate_ir.SetExec(candidate_ir.Imm1(true));
    const auto lane = candidate_ir.LaneId();
    const auto peer = candidate_ir.ReadLane(
        payload, candidate_ir.BitwiseXor(lane, candidate_ir.Imm32(16U)));
    const auto candidate_low = candidate_ir.ReadLane(peer, candidate_ir.Imm32(0U));
    const auto candidate_high = candidate_ir.ReadLane(peer, candidate_ir.Imm32(32U));
    candidate_ir.Reference(candidate_ir.IAdd(candidate_low, candidate_high));

    AppendBlockNode(fixture.program, fixture.block);
    AppendIfNode(fixture.program, outer_condition, blocks.header, blocks.exit);
    AppendBlockNode(fixture.program, blocks.header);
    AppendIfNode(fixture.program, inner_ref, blocks.first_body, blocks.candidate);
    AppendBlockNode(fixture.program, blocks.first_body);
    AppendIfNode(fixture.program, nested_condition, blocks.nested_body, blocks.nested_merge);
    AppendBlockNode(fixture.program, blocks.nested_body);
    AppendEndIfNode(fixture.program, blocks.nested_merge);
    AppendBlockNode(fixture.program, blocks.nested_merge);
    AppendIfNode(fixture.program, second_condition, blocks.second_body, blocks.second_merge);
    AppendBlockNode(fixture.program, blocks.second_body);
    AppendEndIfNode(fixture.program, blocks.second_merge);
    AppendBlockNode(fixture.program, blocks.second_merge);
    AppendEndIfNode(fixture.program, blocks.candidate);
    AppendBlockNode(fixture.program, blocks.candidate);
    AppendEndIfNode(fixture.program, blocks.exit);
    AppendBlockNode(fixture.program, blocks.exit);
    fixture.program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    return blocks;
}

TEST(UniformWaveMergeAdmission, BalancedDivergentDiamondsReconvergeInsideTheUniformOuterBranch) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    const auto blocks = BuildNestedWaveMergeProbe(fixture);

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::Barrier), 4U);
    EXPECT_EQ(CountOpcode(*blocks.header, IR::Opcode::Barrier), 4U);
    EXPECT_EQ(CountOpcode(*blocks.candidate, IR::Opcode::Barrier), 6U);
    EXPECT_FALSE(HasOpcode(*blocks.candidate, IR::Opcode::ReadLane));
    EXPECT_FALSE(HasOpcode(*blocks.candidate, IR::Opcode::LaneId));
    EXPECT_TRUE(HasOpcode(*blocks.candidate, IR::Opcode::Phi));
    EXPECT_TRUE(HasOpcode(*blocks.first_body, IR::Opcode::LaneId));
    EXPECT_EQ(CountOpcode(*blocks.first_body, IR::Opcode::Barrier), 0U);
    EXPECT_EQ(CountOpcode(*blocks.nested_body, IR::Opcode::Barrier), 0U);
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 256U);
}

TEST(UniformWaveMergeAdmission, ASeparateFutureLoopDoesNotInvalidateTheCompletedPrefix) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    const auto blocks = BuildNestedWaveMergeProbe(fixture);
    auto* const loop_header = fixture.pools.block_pool.Create(fixture.pools.inst_pool);
    auto* const loop_body = fixture.pools.block_pool.Create(fixture.pools.inst_pool);
    auto* const after_loop = fixture.pools.block_pool.Create(fixture.pools.inst_pool);
    blocks.exit->AddBranch(loop_header);
    loop_header->AddBranch(loop_body);
    loop_body->AddBranch(loop_header);
    loop_body->AddBranch(after_loop);
    fixture.program.blocks.insert(fixture.program.blocks.end(), {loop_header, loop_body, after_loop});
    fixture.program.syntax_list.pop_back();
    AppendBlockNode(fixture.program, loop_header);
    auto& loop = fixture.program.syntax_list.emplace_back();
    loop.type = IR::AbstractSyntaxNode::Type::Loop;
    loop.data.loop.body = loop_body;
    loop.data.loop.continue_block = loop_body;
    loop.data.loop.merge = after_loop;
    AppendBlockNode(fixture.program, loop_body);
    IR::IREmitter loop_ir{*loop_body};
    loop_ir.Reference(loop_ir.LaneId());
    auto& repeat = fixture.program.syntax_list.emplace_back();
    repeat.type = IR::AbstractSyntaxNode::Type::Repeat;
    repeat.data.repeat.cond = loop_ir.Imm1(false);
    repeat.data.repeat.loop_header = loop_header;
    repeat.data.repeat.merge = after_loop;
    AppendBlockNode(fixture.program, after_loop);
    fixture.program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_EQ(CountOpcode(*blocks.candidate, IR::Opcode::Barrier), 6U);
    EXPECT_FALSE(HasOpcode(*blocks.candidate, IR::Opcode::ReadLane));
    EXPECT_TRUE(HasOpcode(*loop_body, IR::Opcode::LaneId));
    EXPECT_EQ(CountOpcode(*loop_body, IR::Opcode::Barrier), 0U);
}

TEST(UniformWaveMergeAdmission, PartialOrUnrestoredExecDoesNotAdmitTheMerge) {
    for (const bool remove_restore : {false, true}) {
        LaneIdPassFixture fixture{{8U, 8U, 1U}};
        const auto blocks = BuildNestedWaveMergeProbe(fixture);
        auto* const exec = FindOpcode(*blocks.candidate, IR::Opcode::SetExec);
        ASSERT_NE(exec, nullptr);
        if (remove_restore) {
            exec->SetArg(0, IR::Value{});
            exec->ReplaceOpcode(IR::Opcode::Void);
        } else {
            exec->SetArg(0, IR::Value{false});
        }
        Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                            fixture.profile);
        EXPECT_EQ(CountOpcode(*blocks.candidate, IR::Opcode::Barrier), 0U);
        EXPECT_TRUE(HasOpcode(*blocks.candidate, IR::Opcode::ReadLane));
        EXPECT_TRUE(HasOpcode(*blocks.candidate, IR::Opcode::LaneId));
    }
}

TEST(UniformWaveMergeAdmission, SideEntryAndBypassEdgesInvalidateReconvergence) {
    for (const bool side_entry : {false, true}) {
        LaneIdPassFixture fixture{{8U, 8U, 1U}};
        const auto blocks = BuildNestedWaveMergeProbe(fixture);
        if (side_entry) {
            blocks.exit->AddBranch(blocks.candidate);
        } else {
            blocks.nested_body->AddBranch(blocks.exit);
        }
        Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                            fixture.profile);
        EXPECT_EQ(CountOpcode(*blocks.header, IR::Opcode::Barrier), 4U);
        EXPECT_EQ(CountOpcode(*blocks.candidate, IR::Opcode::Barrier), 0U);
        EXPECT_TRUE(HasOpcode(*blocks.candidate, IR::Opcode::ReadLane));
    }
}

TEST(UniformWaveMergeAdmission, UnsupportedOrUnbalancedPrefixRemainsNative) {
    using Type = IR::AbstractSyntaxNode::Type;
    for (const auto blocker : {Type::Loop, Type::Return, Type::Unreachable, Type::Break}) {
        LaneIdPassFixture fixture{{8U, 8U, 1U}};
        const auto blocks = BuildNestedWaveMergeProbe(fixture);
        IR::AbstractSyntaxNode node{};
        node.type = blocker;
        fixture.program.syntax_list.insert(fixture.program.syntax_list.begin() + 7, node);
        Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                            fixture.profile);
        EXPECT_EQ(CountOpcode(*blocks.candidate, IR::Opcode::Barrier), 0U);
        EXPECT_TRUE(HasOpcode(*blocks.candidate, IR::Opcode::ReadLane));
    }
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    const auto blocks = BuildNestedWaveMergeProbe(fixture);
    fixture.program.syntax_list[7].data.end_if.merge = blocks.candidate;
    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);
    EXPECT_EQ(CountOpcode(*blocks.candidate, IR::Opcode::Barrier), 0U);
    EXPECT_TRUE(HasOpcode(*blocks.candidate, IR::Opcode::ReadLane));
}

TEST(UniformWaveMergeAdmission, ARejoinedLaneSelectorCannotEscapeToLaterNativeReads) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    const auto blocks = BuildNestedWaveMergeProbe(fixture);
    auto* const lane = FindOpcode(*blocks.candidate, IR::Opcode::LaneId);
    ASSERT_NE(lane, nullptr);
    IR::IREmitter candidate_ir{*blocks.candidate};
    const IR::U32 selector{candidate_ir.BitwiseXor(IR::U32{lane}, candidate_ir.Imm32(16U))};
    const auto merged_selector = MergeU32(*blocks.exit, blocks.candidate, selector, fixture.block,
                                        candidate_ir.Imm32(0U));
    IR::IREmitter exit_ir{*blocks.exit};
    exit_ir.Reference(exit_ir.ReadLane(exit_ir.Imm32(1U), merged_selector));

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_EQ(CountOpcode(*blocks.candidate, IR::Opcode::Barrier), 0U);
    EXPECT_TRUE(HasOpcode(*blocks.candidate, IR::Opcode::LaneId));
    EXPECT_TRUE(HasOpcode(*blocks.candidate, IR::Opcode::ReadLane));
}

BracketedMergeBlocks BuildTopLevelWaveMergeProbe(LaneIdPassFixture& fixture,
                                                 bool explicit_layout = true) {
    fixture.runtime_info.cs_info.shared_memory_size = 1024U;
    fixture.profile.max_shared_memory_size = 1280U;
    fixture.profile.supports_workgroup_explicit_memory_layout = explicit_layout;
    IR::IREmitter entry_ir{*fixture.block};
    entry_ir.SetExec(entry_ir.Imm1(true));
    const auto x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0U);
    const auto native_entry_lane = entry_ir.LaneId();
    const auto first_condition = entry_ir.ConditionRef(
        entry_ir.IEqual(native_entry_lane, entry_ir.Imm32(0U)));
    entry_ir.SetExec(first_condition);
    const auto second_condition = entry_ir.IEqual(x, entry_ir.Imm32(7U));
    const auto blocks =
        SetTwoSelectionBracketedMergeSyntax(fixture, first_condition, second_condition);
    IR::IREmitter first_ir{*blocks.first_body};
    first_ir.Reference(first_ir.LaneId());
    IR::IREmitter merge_ir{*blocks.merge};
    merge_ir.SetExec(merge_ir.Imm1(true));
    // The original guest shared memory is U64; wave scratch adds U32 and must not enter the
    // storage fallback on a device without explicit mixed-type workgroup layout support.
    merge_ir.Reference(merge_ir.LoadShared(64, false, merge_ir.Imm32(0U)));
    const auto lane = merge_ir.LaneId();
    const auto peer = merge_ir.ReadLane(x, merge_ir.BitwiseXor(lane, merge_ir.Imm32(16U)));
    const auto low = merge_ir.ReadLane(peer, merge_ir.Imm32(0U));
    const auto high = merge_ir.ReadLane(peer, merge_ir.Imm32(32U));
    merge_ir.Reference(merge_ir.IAdd(low, high));
    return blocks;
}

TEST(TopLevelWaveMergeAdmission, FullEntrySelectionRejoinsWithoutAnEntryWaveOperation) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    const auto blocks = BuildTopLevelWaveMergeProbe(fixture);

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_FALSE(HasOpcode(*blocks.merge, IR::Opcode::LaneId));
    EXPECT_FALSE(HasOpcode(*blocks.merge, IR::Opcode::ReadLane));
    EXPECT_EQ(CountOpcode(*blocks.merge, IR::Opcode::Barrier), 6U);
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 256U);
    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::LaneId), 1U);
    EXPECT_EQ(CountOpcode(*blocks.first_body, IR::Opcode::LaneId), 1U);
    EXPECT_EQ(CountOpcode(*fixture.block, IR::Opcode::Barrier), 0U);
    EXPECT_EQ(CountOpcode(*blocks.first_body, IR::Opcode::Barrier), 0U);
    EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::LoadSharedU64));
}

TEST(TopLevelWaveMergeAdmission, MixedGuestSharedMemoryStillRequiresExplicitLayout) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    const auto blocks = BuildTopLevelWaveMergeProbe(fixture, false);

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::LaneId));
    EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::ReadLane));
    EXPECT_EQ(CountOpcode(*blocks.merge, IR::Opcode::Barrier), 0U);
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
}

TEST(TopLevelWaveMergeAdmission, MergeRequiresAnExplicitFullExecRestore) {
    for (const bool remove_restore : {false, true}) {
        SCOPED_TRACE(remove_restore);
        LaneIdPassFixture fixture{{8U, 8U, 1U}};
        const auto blocks = BuildTopLevelWaveMergeProbe(fixture);
        auto* const restore = FindOpcode(*blocks.merge, IR::Opcode::SetExec);
        ASSERT_NE(restore, nullptr);
        if (remove_restore) {
            restore->SetArg(0, IR::Value{});
            restore->ReplaceOpcode(IR::Opcode::Void);
        } else {
            restore->SetArg(0, IR::Value{false});
        }

        Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                            fixture.profile);

        EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::ReadLane));
        EXPECT_EQ(CountOpcode(*blocks.merge, IR::Opcode::Barrier), 0U);
    }
}

TEST(TopLevelWaveMergeAdmission, EntryBackedgeOrEarlyExitInvalidatesThePrefix) {
    for (const bool backedge : {false, true}) {
        SCOPED_TRACE(backedge);
        LaneIdPassFixture fixture{{8U, 8U, 1U}};
        const auto blocks = BuildTopLevelWaveMergeProbe(fixture);
        if (backedge) {
            blocks.exit->AddBranch(fixture.block);
        } else {
            blocks.first_body->AddBranch(blocks.exit);
        }

        Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                            fixture.profile);

        EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::ReadLane));
        EXPECT_EQ(CountOpcode(*blocks.merge, IR::Opcode::Barrier), 0U);
    }
}

TEST(TopLevelWaveMergeAdmission, AnIncomingNativeLaneSelectorIsNotAGuestWaveSelector) {
    for (const bool use_phi : {false, true}) {
        SCOPED_TRACE(use_phi);
        LaneIdPassFixture fixture{{8U, 8U, 1U}};
        const auto blocks = BuildTopLevelWaveMergeProbe(fixture);
        auto* const native_lane = FindOpcode(*fixture.block, IR::Opcode::LaneId);
        ASSERT_NE(native_lane, nullptr);
        IR::IREmitter entry_ir{*fixture.block};
        const IR::U32 forwarded{entry_ir.BitwiseXor(IR::U32{native_lane}, entry_ir.Imm32(16U))};
        const IR::U32 selector = use_phi
                                     ? MergeU32(*blocks.merge, blocks.first_body, forwarded,
                                                fixture.block, forwarded)
                                     : forwarded;
        auto* const read = FindOpcode(*blocks.merge, IR::Opcode::ReadLane);
        ASSERT_NE(read, nullptr);
        read->SetArg(1, selector);

        Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                            fixture.profile);

        EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::ReadLane));
        EXPECT_EQ(CountOpcode(*blocks.merge, IR::Opcode::Barrier), 0U);
        EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
    }
}

TEST(TopLevelWaveMergeAdmission, IncomingNativeLaneDataAndBallotPredicatesRemainConservative) {
    for (const bool ballot : {false, true}) {
        SCOPED_TRACE(ballot);
        LaneIdPassFixture fixture{{8U, 8U, 1U}};
        const auto blocks = BuildTopLevelWaveMergeProbe(fixture);
        auto* const native_lane = FindOpcode(*fixture.block, IR::Opcode::LaneId);
        ASSERT_NE(native_lane, nullptr);
        if (ballot) {
            IR::IREmitter merge_ir{*blocks.merge};
            merge_ir.Reference(merge_ir.Ballot(
                merge_ir.IEqual(IR::U32{native_lane}, merge_ir.Imm32(0U))));
        } else {
            auto* const read = FindOpcode(*blocks.merge, IR::Opcode::ReadLane);
            ASSERT_NE(read, nullptr);
            read->SetArg(0, IR::Value{native_lane});
        }

        Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                            fixture.profile);

        EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::ReadLane));
        EXPECT_EQ(CountOpcode(*blocks.merge, IR::Opcode::Barrier), 0U);
        EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
    }
}

TEST(TopLevelWaveMergeAdmission, IncomingUnselectedWaveResultsAreNotGuestWaveValues) {
    for (const u32 kind : {0U, 1U, 2U}) {
        SCOPED_TRACE(kind);
        LaneIdPassFixture fixture{{8U, 8U, 1U}};
        const auto blocks = BuildTopLevelWaveMergeProbe(fixture);
        IR::IREmitter entry_ir{*fixture.block};
        const auto x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0U);
        const auto y = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1U);
        const IR::U32 invocation{entry_ir.IAdd(x, entry_ir.IMul(y, entry_ir.Imm32(8U)))};
        IR::U32 value;
        if (kind == 0U) {
            value = entry_ir.ReadFirstLane(invocation);
        } else if (kind == 1U) {
            const auto read = entry_ir.ReadLane(invocation, entry_ir.Imm32(0U));
            auto* const identity = &*fixture.block->PrependNewInst(
                fixture.block->end(), IR::Opcode::Identity, {read});
            value = IR::U32{identity};
        } else {
            value = IR::U32{entry_ir.CompositeExtract(
                entry_ir.Ballot(entry_ir.IEqual(invocation, entry_ir.Imm32(0U))), 0U)};
        }
        auto* const read = FindOpcode(*blocks.merge, IR::Opcode::ReadLane);
        ASSERT_NE(read, nullptr);
        read->SetArg(0, value);

        Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                            fixture.profile);

        EXPECT_TRUE(HasOpcode(*blocks.merge, IR::Opcode::ReadLane));
        EXPECT_EQ(CountOpcode(*blocks.merge, IR::Opcode::Barrier), 0U);
        EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
    }
}

TEST(UniformWaveBodyAdmission, ASelectedEntryReadMaySupplyTheBodyReadValueAndSelector) {
    LaneIdPassFixture fixture{{8U, 8U, 1U}};
    auto* const body = BuildFirstUniformBodyProbe(
        fixture, FirstBodyCondition::FullWaveReadResult, FirstBodyExec::ExplicitFull);
    auto* const entry_read = FindOpcode(*fixture.block, IR::Opcode::ReadLane);
    auto* const body_read = FindOpcode(*body, IR::Opcode::ReadLane);
    ASSERT_NE(entry_read, nullptr);
    ASSERT_NE(body_read, nullptr);
    body_read->SetArg(0, IR::Value{entry_read});
    body_read->SetArg(1, IR::Value{entry_read});

    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime_info,
                                                        fixture.profile);

    EXPECT_FALSE(HasOpcode(*fixture.block, IR::Opcode::ReadLane));
    EXPECT_FALSE(HasOpcode(*body, IR::Opcode::ReadLane));
    EXPECT_EQ(CountOpcode(*body, IR::Opcode::Barrier), 6U);
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 256U);
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
    ir.SetExec(ir.Imm1(false));
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

} // namespace
