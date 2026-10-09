// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <gtest/gtest.h>

#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"

namespace {
using namespace Shader;

struct CompareBallotPrefixFixture {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    RuntimeInfo runtime{};
    Profile profile{};
    IR::Block* block{};

    CompareBallotPrefixFixture() {
        info.stage = Stage::Compute;
        info.l_stage = LogicalStage::Compute;
        runtime.Initialize(Stage::Compute);
        runtime.cs_info.workgroup_size = {64, 1, 1};
        profile.subgroup_size = 32;
        profile.max_shared_memory_size = 32 * 1024;
        block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        auto& node = program.syntax_list.emplace_back();
        node.type = IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
        program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
        program.post_order_blocks = IR::PostOrder(program.syntax_list.front());
    }

    void Run() {
        Optimization::LowerLaneIdToLocalInvocationIndexPass(program, runtime, profile);
    }

    size_t Count(IR::Opcode opcode) const {
        return std::ranges::count_if(
            *block, [=](const IR::Inst& inst) { return inst.GetOpcode() == opcode; });
    }
};

TEST(CompareBallotPrefix, FullEntryBallotIsNotRejectedByALaterCmpxExecWrite) {
    CompareBallotPrefixFixture fixture;
    IR::IREmitter ir{*fixture.block};
    ir.SetExec(ir.Imm1(true));
    const auto predicate =
        ir.ILessThan(ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0), ir.Imm32(40U), false);
    ir.Reference(ir.Ballot(predicate));
    ir.SetExec(predicate);
    fixture.Run();
    EXPECT_EQ(fixture.Count(IR::Opcode::Ballot), 0U);
    EXPECT_GT(fixture.Count(IR::Opcode::Barrier), 0U);
    bool changed_exec{};
    for (const auto& inst : *fixture.block) {
        if (inst.GetOpcode() == IR::Opcode::SetExec && !inst.Arg(0).IsImmediate()) {
            changed_exec = true;
        }
        if (inst.GetOpcode() == IR::Opcode::Barrier) {
            EXPECT_FALSE(changed_exec) << "No generated workgroup barrier may follow partial EXEC";
        }
    }
}

TEST(CompareBallotPrefix, PartialEntryBallotRemainsNativeWithoutAddingBarriers) {
    CompareBallotPrefixFixture fixture;
    IR::IREmitter ir{*fixture.block};
    const auto predicate =
        ir.ILessThan(ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0), ir.Imm32(4U), false);
    ir.SetExec(predicate);
    ir.Reference(ir.Ballot(predicate));
    fixture.Run();
    EXPECT_EQ(fixture.Count(IR::Opcode::Ballot), 1U);
    EXPECT_EQ(fixture.Count(IR::Opcode::Barrier), 0U);
}

TEST(CompareBallotPrefix, NoExecMutationKeepsExistingFullEntryBallotLowering) {
    CompareBallotPrefixFixture fixture;
    IR::IREmitter ir{*fixture.block};
    ir.SetExec(ir.Imm1(true));
    ir.Reference(ir.Ballot(ir.Imm1(true)));
    fixture.Run();
    EXPECT_EQ(fixture.Count(IR::Opcode::Ballot), 0U);
    EXPECT_GT(fixture.Count(IR::Opcode::Barrier), 0U);
}
} // namespace
