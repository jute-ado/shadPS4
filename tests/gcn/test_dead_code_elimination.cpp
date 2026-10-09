// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"

namespace {
using namespace Shader;

struct DeadCycleFixture {
    DeadCycleFixture() : program{info} {
        info.stage = Stage::Compute;
        info.l_stage = LogicalStage::Compute;
        entry = pools.block_pool.Create(pools.inst_pool);
        loop = pools.block_pool.Create(pools.inst_pool);
        exit = pools.block_pool.Create(pools.inst_pool);
        entry->AddBranch(loop);
        loop->AddBranch(loop);
        loop->AddBranch(exit);
        program.blocks = {entry, loop, exit};
        program.post_order_blocks = {exit, loop, entry};
        auto& entry_node = program.syntax_list.emplace_back();
        entry_node.type = IR::AbstractSyntaxNode::Type::Block;
        entry_node.data.block = entry;
        auto& loop_node = program.syntax_list.emplace_back();
        loop_node.type = IR::AbstractSyntaxNode::Type::Loop;
        loop_node.data.loop = {loop, loop, exit};
        auto& body_node = program.syntax_list.emplace_back();
        body_node.type = IR::AbstractSyntaxNode::Type::Block;
        body_node.data.block = loop;
        auto& repeat = program.syntax_list.emplace_back();
        repeat.type = IR::AbstractSyntaxNode::Type::Repeat;
        repeat.data.repeat = {IR::U1{IR::Value{false}}, loop, exit};
        auto& exit_node = program.syntax_list.emplace_back();
        exit_node.type = IR::AbstractSyntaxNode::Type::Block;
        exit_node.data.block = exit;
        program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
        IR::IREmitter{*exit}.Epilogue();
    }

    IR::U32 AddSpillCycle() {
        entry->AppendNewInst(IR::Opcode::UndefU32, {});
        const IR::U32 undefined{IR::Value{&entry->back()}};
        IR::IREmitter first{*entry};
        initial = first.WriteLane(undefined, first.Imm32(55), first.Imm32(0)).Inst();
        phi = &*loop->PrependNewInst(loop->begin(), IR::Opcode::Phi, {},
                                     static_cast<u32>(IR::Type::U32));
        phi->AddPhiOperand(entry, IR::Value{initial});
        IR::IREmitter body{*loop};
        backedge = body.WriteLane(IR::U32{IR::Value{phi}}, body.Imm32(66), body.Imm32(0)).Inst();
        phi->AddPhiOperand(loop, IR::Value{backedge});
        return IR::U32{IR::Value{phi}};
    }

    size_t Count(IR::Opcode opcode) const {
        size_t result = 0;
        for (auto* block : program.blocks) {
            result += std::ranges::count_if(
                *block, [&](const auto& inst) { return inst.GetOpcode() == opcode; });
        }
        return result;
    }

    Info info{};
    Pools pools{};
    IR::Program program;
    IR::Block *entry{}, *loop{}, *exit{};
    IR::Inst *initial{}, *phi{}, *backedge{};
};

TEST(DeadCodeElimination, RemovesPureCrossBlockPhiWriteLaneCycleInOnePass) {
    DeadCycleFixture fixture;
    fixture.AddSpillCycle();
    ASSERT_TRUE(fixture.phi->HasUses());
    ASSERT_TRUE(fixture.backedge->HasUses());
    Optimization::DeadCodeEliminationPass(fixture.program);
    EXPECT_EQ(fixture.Count(IR::Opcode::WriteLane), 0U);
    EXPECT_EQ(fixture.Count(IR::Opcode::Phi), 0U);
    EXPECT_EQ(fixture.Count(IR::Opcode::UndefU32), 0U);
    EXPECT_EQ(fixture.Count(IR::Opcode::Epilogue), 1U);
    for (const auto* inst : {fixture.initial, fixture.phi, fixture.backedge}) {
        EXPECT_FALSE(inst->HasUses());
    }
    Optimization::DeadCodeEliminationPass(fixture.program);
    EXPECT_EQ(fixture.Count(IR::Opcode::Epilogue), 1U);
}

TEST(DeadCodeElimination, PreservesCrossBlockCycleFeedingAStore) {
    DeadCycleFixture fixture;
    const auto value = fixture.AddSpillCycle();
    IR::IREmitter ir{*fixture.exit};
    ir.StoreBufferU32(1, ir.Imm32(0), ir.Imm32(0), value, {});
    Optimization::DeadCodeEliminationPass(fixture.program);
    EXPECT_EQ(fixture.Count(IR::Opcode::WriteLane), 2U);
    EXPECT_EQ(fixture.Count(IR::Opcode::Phi), 1U);
    EXPECT_EQ(fixture.Count(IR::Opcode::UndefU32), 1U);
    EXPECT_EQ(fixture.Count(IR::Opcode::StoreBufferU32), 1U);
    EXPECT_EQ(fixture.phi->Arg(1).Inst(), fixture.backedge);
}

class SyntaxOnlyDceCondition : public testing::TestWithParam<IR::AbstractSyntaxNode::Type> {};

TEST_P(SyntaxOnlyDceCondition, PreservesConditionAndCycleWithoutInstructionUse) {
    DeadCycleFixture fixture;
    const auto value = fixture.AddSpillCycle();
    IR::IREmitter ir{*fixture.loop};
    const auto condition = ir.IEqual(value, ir.Imm32(7));
    ASSERT_FALSE(condition.Inst()->HasUses());
    auto& node = fixture.program.syntax_list.emplace_back();
    node.type = GetParam();
    switch (GetParam()) {
    case IR::AbstractSyntaxNode::Type::If:
        node.data.if_node = {condition, fixture.loop, fixture.exit};
        break;
    case IR::AbstractSyntaxNode::Type::Repeat:
        node.data.repeat = {condition, fixture.loop, fixture.exit};
        break;
    case IR::AbstractSyntaxNode::Type::Break:
        node.data.break_node = {condition, fixture.exit, fixture.loop};
        break;
    default:
        FAIL();
    }
    Optimization::DeadCodeEliminationPass(fixture.program);
    EXPECT_EQ(condition.Inst()->GetOpcode(), IR::Opcode::IEqual32);
    EXPECT_EQ(fixture.Count(IR::Opcode::WriteLane), 2U);
    EXPECT_EQ(fixture.Count(IR::Opcode::Phi), 1U);
}

INSTANTIATE_TEST_SUITE_P(AllSyntaxConditions, SyntaxOnlyDceCondition,
                         testing::Values(IR::AbstractSyntaxNode::Type::If,
                                         IR::AbstractSyntaxNode::Type::Repeat,
                                         IR::AbstractSyntaxNode::Type::Break));

TEST(DeadCodeElimination, PreservesIdentityToImmediateEvenWithoutTrackedUseCount) {
    DeadCycleFixture fixture;
    fixture.entry->AppendNewInst(IR::Opcode::Identity, {IR::Value{u32{123}}});
    auto* identity = &fixture.entry->back();
    IR::IREmitter ir{*fixture.exit};
    ir.StoreBufferU32(1, ir.Imm32(0), ir.Imm32(0), IR::U32{IR::Value{identity}}, {});
    ASSERT_FALSE(identity->HasUses()); // IsImmediate transparently unwraps Identity.
    Optimization::DeadCodeEliminationPass(fixture.program);
    ASSERT_EQ(identity->GetOpcode(), IR::Opcode::Identity);
    EXPECT_EQ(identity->Arg(0).U32(), 123U);
    EXPECT_EQ(fixture.exit->back().Arg(2).Inst(), identity);
}

TEST(DeadCodeElimination, PreservesIdentityChainToLiveCycle) {
    DeadCycleFixture fixture;
    const auto value = fixture.AddSpillCycle();
    fixture.exit->AppendNewInst(IR::Opcode::Identity, {value});
    auto* identity = &fixture.exit->back();
    IR::IREmitter{*fixture.exit}.Reference(IR::Value{identity});
    Optimization::DeadCodeEliminationPass(fixture.program);
    EXPECT_EQ(identity->GetOpcode(), IR::Opcode::Identity);
    EXPECT_EQ(fixture.Count(IR::Opcode::WriteLane), 2U);
    EXPECT_EQ(fixture.Count(IR::Opcode::Phi), 1U);
}

bool CleanupDeadConstantIdentity(bool reverse_blocks, bool cross_block, bool phi_consumer) {
    DeadCycleFixture fixture;
    fixture.entry->AppendNewInst(IR::Opcode::Identity, {IR::Value{u32{91}}});
    auto* first = &fixture.entry->back();
    fixture.entry->AppendNewInst(IR::Opcode::Identity, {IR::Value{first}});
    auto* second = &fixture.entry->back();
    auto* block = cross_block ? fixture.loop : fixture.entry;
    if (phi_consumer) {
        auto* phi = &*block->PrependNewInst(block->begin(), IR::Opcode::Phi, {},
                                            static_cast<u32>(IR::Type::U32));
        phi->AddPhiOperand(fixture.entry, IR::Value{second});
    } else {
        IR::IREmitter ir{*block};
        const auto unused = ir.IAdd(IR::U32{IR::Value{second}}, ir.Imm32(1));
        if (!unused.Inst()->Arg(0).IsIdentity()) {
            return false;
        }
    }
    if (first->HasUses() || second->HasUses()) {
        return false;
    }
    if (reverse_blocks) {
        std::ranges::reverse(fixture.program.blocks);
    }
    Optimization::DeadCodeEliminationPass(fixture.program);
    return fixture.Count(IR::Opcode::Identity) == 0 && fixture.Count(IR::Opcode::IAdd32) == 0 &&
           fixture.Count(IR::Opcode::Phi) == 0 && !first->HasUses() && !second->HasUses();
}

class DeadConstantIdentityOrder : public testing::TestWithParam<bool> {};

TEST_P(DeadConstantIdentityOrder, RemovesCrossBlockDeadChainWithoutUntrackedUseRemoval) {
    // The actual IR use-list checks are enabled for microinstruction.cpp. Keep failures in a
    // subprocess: producer-first invalidation used to assert on an untracked UndoUse edge.
    EXPECT_EXIT(std::_Exit(CleanupDeadConstantIdentity(GetParam(), true, false) ? 0 : 2),
                testing::ExitedWithCode(0), "");
}

TEST_P(DeadConstantIdentityOrder, RemovesDeadConstantIdentityPhiOperand) {
    EXPECT_EXIT(std::_Exit(CleanupDeadConstantIdentity(GetParam(), true, true) ? 0 : 2),
                testing::ExitedWithCode(0), "");
}

INSTANTIATE_TEST_SUITE_P(ProducerFirstAndConsumerFirst, DeadConstantIdentityOrder, testing::Bool());

TEST(DeadCodeElimination, RemovesSameBlockDeadConstantIdentityAndArithmetic) {
    EXPECT_EXIT(std::_Exit(CleanupDeadConstantIdentity(false, false, false) ? 0 : 2),
                testing::ExitedWithCode(0), "");
}

TEST(DeadCodeElimination, RemovesDeadIdentityIntoTrackedPhiCycleWithoutStaleUses) {
    DeadCycleFixture fixture;
    fixture.AddSpillCycle();
    fixture.loop->AppendNewInst(IR::Opcode::Identity, {IR::Value{fixture.backedge}});
    auto* identity = &fixture.loop->back();
    fixture.phi->SetArg(1, IR::Value{identity});
    ASSERT_TRUE(identity->HasUses());
    ASSERT_TRUE(fixture.backedge->HasUses());
    Optimization::DeadCodeEliminationPass(fixture.program);
    EXPECT_EQ(fixture.Count(IR::Opcode::Identity), 0U);
    EXPECT_EQ(fixture.Count(IR::Opcode::Phi), 0U);
    EXPECT_EQ(fixture.Count(IR::Opcode::WriteLane), 0U);
    EXPECT_FALSE(identity->HasUses());
    EXPECT_FALSE(fixture.phi->HasUses());
    EXPECT_FALSE(fixture.backedge->HasUses());
}

TEST(DeadCodeElimination, KeepsLiveIdentityAndSyntaxWhileRemovingDeadImmediateConsumer) {
    DeadCycleFixture fixture;
    fixture.entry->AppendNewInst(IR::Opcode::Identity, {IR::Value{false}});
    auto* identity = &fixture.entry->back();
    auto& repeat = fixture.program.syntax_list[3];
    ASSERT_EQ(repeat.type, IR::AbstractSyntaxNode::Type::Repeat);
    repeat.data.repeat.cond = IR::U1{IR::Value{identity}};
    IR::IREmitter ir{*fixture.loop};
    static_cast<void>(ir.LogicalNot(IR::U1{IR::Value{identity}}));
    ASSERT_FALSE(identity->HasUses());
    Optimization::DeadCodeEliminationPass(fixture.program);
    EXPECT_EQ(fixture.Count(IR::Opcode::Identity), 1U);
    EXPECT_EQ(fixture.Count(IR::Opcode::LogicalNot), 0U);
    EXPECT_EQ(repeat.data.repeat.cond.Inst(), identity);
}

TEST(DeadCodeElimination, RetainsSideEffectsEvenInDisconnectedBlock) {
    DeadCycleFixture fixture;
    auto* disconnected = fixture.pools.block_pool.Create(fixture.pools.inst_pool);
    fixture.program.blocks.push_back(disconnected);
    IR::IREmitter ir{*disconnected};
    const auto value = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    ir.WriteShared(32, value, ir.Imm32(0));
    ir.Barrier();
    Optimization::DeadCodeEliminationPass(fixture.program);
    EXPECT_EQ(fixture.Count(IR::Opcode::WriteSharedU32), 1U);
    EXPECT_EQ(fixture.Count(IR::Opcode::Barrier), 1U);
    EXPECT_EQ(fixture.Count(IR::Opcode::GetAttributeU32), 1U);
}

TEST(DeadCodeElimination, RemovesDeadSpillCycleBeforeOtherwiseSafeLaneIdLowering) {
    DeadCycleFixture fixture;
    fixture.AddSpillCycle();
    IR::IREmitter ir{*fixture.entry};
    ir.Reference(ir.LaneId());
    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Compute);
    runtime.cs_info.workgroup_size = {8, 8, 1};
    Profile profile{};
    profile.subgroup_size = 32;
    Optimization::DeadCodeEliminationPass(fixture.program);
    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, runtime, profile);
    EXPECT_EQ(fixture.Count(IR::Opcode::WriteLane), 0U);
    EXPECT_EQ(fixture.Count(IR::Opcode::LaneId), 0U);
    EXPECT_GT(fixture.Count(IR::Opcode::GetAttributeU32), 0U);
}

TEST(DeadCodeElimination, TrulyLiveWriteLaneStillBlocksUnsafeLaneIdLowering) {
    DeadCycleFixture fixture;
    const auto value = fixture.AddSpillCycle();
    IR::IREmitter ir{*fixture.entry};
    ir.Reference(ir.LaneId());
    IR::IREmitter{*fixture.exit}.Reference(value);
    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Compute);
    runtime.cs_info.workgroup_size = {8, 8, 1};
    Profile profile{};
    profile.subgroup_size = 32;
    Optimization::DeadCodeEliminationPass(fixture.program);
    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, runtime, profile);
    EXPECT_EQ(fixture.Count(IR::Opcode::WriteLane), 2U);
    EXPECT_EQ(fixture.Count(IR::Opcode::LaneId), 1U);
}

void RunEarlyCleanup(IR::Program& program) {
    Optimization::DeadCodeEliminationPass(program, true);
}

void LowerEntryLanes(DeadCycleFixture& fixture) {
    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Compute);
    runtime.cs_info.workgroup_size = {8, 8, 1};
    Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 32 * 1024;
    Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, runtime, profile);
}

class EarlyDceExecState : public testing::TestWithParam<bool> {};

TEST_P(EarlyDceExecState, PreservesFalseOrDynamicMaskBeforeLaneAnalysis) {
    DeadCycleFixture fixture;
    IR::IREmitter ir{*fixture.entry};
    const auto mask =
        GetParam() ? ir.IEqual(ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0), ir.Imm32(0))
                   : ir.Imm1(false);
    ir.SetExec(mask);
    ir.Reference(ir.LaneId());
    ir.Reference(ir.ReadFirstLane(ir.Imm32(1)));
    RunEarlyCleanup(fixture.program);
    EXPECT_EQ(fixture.Count(IR::Opcode::SetExec), 1U);
    LowerEntryLanes(fixture);
    EXPECT_EQ(fixture.Count(IR::Opcode::LaneId), 1U);
    EXPECT_EQ(fixture.Count(IR::Opcode::ReadFirstLane), 1U);
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
    Optimization::DeadCodeEliminationPass(fixture.program);
    EXPECT_EQ(fixture.Count(IR::Opcode::SetExec), 0U);
}

INSTANTIATE_TEST_SUITE_P(FalseAndDynamicMasks, EarlyDceExecState, testing::Bool());

TEST(DeadCodeElimination, EarlyCleanupDoesNotLetLaterFullMaskRepairEarlierMaskedRead) {
    DeadCycleFixture fixture;
    IR::IREmitter ir{*fixture.entry};
    ir.SetExec(ir.Imm1(false));
    ir.Reference(ir.LaneId());
    ir.Reference(ir.ReadFirstLane(ir.Imm32(1)));
    ir.SetExec(ir.Imm1(true));
    RunEarlyCleanup(fixture.program);
    EXPECT_EQ(fixture.Count(IR::Opcode::SetExec), 2U);
    LowerEntryLanes(fixture);
    EXPECT_EQ(fixture.Count(IR::Opcode::ReadFirstLane), 1U);
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
}

TEST(DeadCodeElimination, EarlyCleanupKeepsProvenFullEntryThenFinalCleanupRemovesMarkers) {
    DeadCycleFixture fixture;
    IR::IREmitter ir{*fixture.entry};
    ir.SetExec(ir.Imm1(true));
    // Keep the entry lane live through the selected wave operation, rather than exposing the
    // unconverted lane directly through an observable terminal outside the confined proof.
    ir.Reference(ir.ReadFirstLane(ir.LaneId()));
    ir.SetExec(ir.Imm1(false));
    RunEarlyCleanup(fixture.program);
    EXPECT_EQ(fixture.Count(IR::Opcode::SetExec), 2U);
    LowerEntryLanes(fixture);
    EXPECT_EQ(fixture.Count(IR::Opcode::LaneId), 0U);
    EXPECT_EQ(fixture.Count(IR::Opcode::ReadFirstLane), 0U);
    EXPECT_GT(fixture.info.workgroup_scratch_memory_size, 0U);
    Optimization::DeadCodeEliminationPass(fixture.program);
    EXPECT_EQ(fixture.Count(IR::Opcode::SetExec), 0U);
}

TEST(DeadCodeElimination, EarlyCleanupRetainsUnusedGetExecUntilFinalCleanup) {
    DeadCycleFixture fixture;
    static_cast<void>(IR::IREmitter{*fixture.entry}.GetExec());
    RunEarlyCleanup(fixture.program);
    EXPECT_EQ(fixture.Count(IR::Opcode::GetExec), 1U);
    Optimization::DeadCodeEliminationPass(fixture.program);
    EXPECT_EQ(fixture.Count(IR::Opcode::GetExec), 0U);
}

TEST(DeadCodeElimination, ProductionCleanupFollowsResourceAdmissionAndPrecedesLaneAnalysis) {
    const auto path = std::filesystem::path{__FILE__}.parent_path().parent_path().parent_path() /
                      "src/shader_recompiler/recompiler.cpp";
    std::ifstream input{path, std::ios::binary};
    ASSERT_TRUE(input.is_open());
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto resources = source.find("Shader::Optimization::ResourceTrackingPass(program");
    ASSERT_NE(resources, std::string::npos);
    const auto cleanup =
        source.find("Shader::Optimization::DeadCodeEliminationPass(program, true)", resources);
    const auto lower = source.find(
        "Shader::Optimization::LowerLaneIdToLocalInvocationIndexPass(program", resources);
    ASSERT_NE(cleanup, std::string::npos);
    ASSERT_NE(lower, std::string::npos);
    EXPECT_LT(cleanup, lower);
    const auto final_cleanup =
        source.find("Shader::Optimization::DeadCodeEliminationPass(program)", lower);
    EXPECT_NE(final_cleanup, std::string::npos);
}
} // namespace
