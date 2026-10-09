// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"

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

TEST(Wave64MaskLoweringPass, UsesOneWorkgroupWaveForLaneIdsAndBothBallotWords) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U32 lane = ir.LaneId();
    const Shader::IR::U1 predicate = ir.INotEqual(lane, ir.Imm32(63));
    const Shader::IR::Value ballot = ir.Ballot(predicate);
    ir.Reference(ir.CompositeExtract(ballot, 1));
    ir.Reference(ir.MaterializeScalarMaskWord(ir.Imm1(true), predicate, ir.Imm32(7), 1));
    ir.Epilogue();

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {4, 2, 8};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;

    Shader::Optimization::Wave64MaskLoweringPass(program, runtime_info, profile);

    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::LaneId), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Ballot), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::MaterializeScalarMaskWord), 0U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::SharedAtomicAnd32), 2U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::SharedAtomicOr32), 2U);
    EXPECT_EQ(CountOpcode(program, Shader::IR::Opcode::Barrier), 6U);
    EXPECT_EQ(runtime_info.cs_info.shared_memory_size, 8U);
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
