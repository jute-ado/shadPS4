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

size_t CountBarriers(const Shader::IR::Block& block) {
    return std::ranges::count_if(block.Instructions(), [](const Shader::IR::Inst& inst) {
        return inst.GetOpcode() == Shader::IR::Opcode::Barrier;
    });
}

size_t CountInsertedBarriers(u32 workgroup_size, u32 shared_memory_size = 256,
                             bool needs_lds_barriers = true) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    Shader::IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter ir{*block};
    ir.WriteShared(32, ir.Imm32(42U), ir.Imm32(0U));
    ir.Epilogue();

    auto& block_node = program.syntax_list.emplace_back();
    block_node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
    block_node.data.block = block;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.shared_memory_size = shared_memory_size;
    runtime_info.cs_info.workgroup_size = {workgroup_size, 1, 1};

    Shader::Profile profile{};
    profile.needs_lds_barriers = needs_lds_barriers;
    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);

    return CountBarriers(*block);
}

struct NestedBarrierLocations {
    size_t inner_merge;
    size_t outer_merge;
};

NestedBarrierLocations BarriersForMaskedInvocationBranch(const std::array<u32, 3>& workgroup_size) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    Shader::IR::Block* const entry = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter entry_ir{*entry};
    const Shader::IR::U32 invocation_x =
        entry_ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U32 invocation_y =
        entry_ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 1);
    const Shader::IR::U32 invocation_z =
        entry_ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 2);
    const Shader::IR::U32 invocation_xy =
        entry_ir.IAdd(invocation_x, entry_ir.IMul(invocation_y, entry_ir.Imm32(workgroup_size[0])));
    const Shader::IR::U32 invocation = entry_ir.IAdd(
        invocation_xy,
        entry_ir.IMul(invocation_z, entry_ir.Imm32(workgroup_size[0] * workgroup_size[1])));
    const Shader::IR::U32 wave_base = entry_ir.BitwiseAnd(invocation, entry_ir.Imm32(~63U));
    const Shader::IR::U32 shared_value{entry_ir.LoadShared(32, false, wave_base)};
    const Shader::IR::U1 outer_cond =
        entry_ir.ConditionRef(entry_ir.IEqual(shared_value, entry_ir.Imm32(0U)));
    entry_ir.Epilogue();

    Shader::IR::Block* const outer_body = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter outer_ir{*outer_body};
    const Shader::IR::U32 inner_invocation =
        outer_ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U1 inner_cond =
        outer_ir.ConditionRef(outer_ir.IEqual(inner_invocation, outer_ir.Imm32(0U)));
    outer_ir.Epilogue();

    Shader::IR::Block* const inner_body = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter inner_ir{*inner_body};
    static_cast<void>(inner_ir.SharedAtomicOr(inner_ir.Imm32(0U), inner_ir.Imm32(1U), false));
    inner_ir.Epilogue();

    Shader::IR::Block* const inner_merge = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter{*inner_merge}.Epilogue();
    Shader::IR::Block* const outer_merge = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter{*outer_merge}.Epilogue();

    auto add_block = [&](Shader::IR::Block* block) {
        auto& node = program.syntax_list.emplace_back();
        node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
    };
    auto add_if = [&](const Shader::IR::U1& cond, Shader::IR::Block* body,
                      Shader::IR::Block* merge) {
        auto& node = program.syntax_list.emplace_back();
        node.type = Shader::IR::AbstractSyntaxNode::Type::If;
        node.data.if_node = {cond, body, merge};
    };
    auto add_end_if = [&](Shader::IR::Block* merge) {
        auto& node = program.syntax_list.emplace_back();
        node.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
        node.data.end_if.merge = merge;
    };

    add_block(entry);
    add_if(outer_cond, outer_body, outer_merge);
    add_block(outer_body);
    add_if(inner_cond, inner_body, inner_merge);
    add_block(inner_body);
    add_end_if(inner_merge);
    add_block(inner_merge);
    add_end_if(outer_merge);
    add_block(outer_merge);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.shared_memory_size = 256;
    runtime_info.cs_info.workgroup_size = workgroup_size;

    Shader::Profile profile{};
    profile.needs_lds_barriers = true;
    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);

    return {
        .inner_merge = CountBarriers(*inner_merge),
        .outer_merge = CountBarriers(*outer_merge),
    };
}

struct LoopBarrierLocations {
    size_t inner_merge;
    size_t loop_merge;
};

LoopBarrierLocations BarriersForNestedBranchInLoop(bool divergent_repeat) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    Shader::IR::Block* const header = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter{*header}.Epilogue();

    Shader::IR::Block* const loop_body = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter body_ir{*loop_body};
    const Shader::IR::U32 invocation =
        body_ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U1 inner_cond =
        body_ir.ConditionRef(body_ir.IEqual(invocation, body_ir.Imm32(0U)));
    body_ir.Epilogue();

    Shader::IR::Block* const inner_body = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter inner_ir{*inner_body};
    static_cast<void>(inner_ir.SharedAtomicOr(inner_ir.Imm32(0U), inner_ir.Imm32(1U), false));
    inner_ir.Epilogue();

    Shader::IR::Block* const inner_merge = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter{*inner_merge}.Epilogue();

    Shader::IR::Block* const continue_block = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter continue_ir{*continue_block};
    const Shader::IR::U32 repeat_invocation =
        continue_ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U1 repeat_cond = continue_ir.ConditionRef(
        divergent_repeat ? continue_ir.IEqual(repeat_invocation, continue_ir.Imm32(0U))
                         : continue_ir.Imm1(false));
    continue_ir.Epilogue();

    Shader::IR::Block* const loop_merge = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter{*loop_merge}.Epilogue();

    auto add_block = [&](Shader::IR::Block* block) {
        auto& node = program.syntax_list.emplace_back();
        node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
    };

    add_block(header);
    auto& loop = program.syntax_list.emplace_back();
    loop.type = Shader::IR::AbstractSyntaxNode::Type::Loop;
    loop.data.loop = {
        .body = loop_body,
        .continue_block = continue_block,
        .merge = loop_merge,
    };
    add_block(loop_body);
    auto& if_node = program.syntax_list.emplace_back();
    if_node.type = Shader::IR::AbstractSyntaxNode::Type::If;
    if_node.data.if_node = {
        .cond = inner_cond,
        .body = inner_body,
        .merge = inner_merge,
    };
    add_block(inner_body);
    auto& end_if = program.syntax_list.emplace_back();
    end_if.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
    end_if.data.end_if.merge = inner_merge;
    add_block(inner_merge);
    add_block(continue_block);
    auto& repeat = program.syntax_list.emplace_back();
    repeat.type = Shader::IR::AbstractSyntaxNode::Type::Repeat;
    repeat.data.repeat = {
        .cond = repeat_cond,
        .loop_header = header,
        .merge = loop_merge,
    };
    add_block(loop_merge);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.shared_memory_size = 256;
    runtime_info.cs_info.workgroup_size = {64, 1, 1};

    Shader::Profile profile{};
    profile.needs_lds_barriers = true;
    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);

    return {
        .inner_merge = CountBarriers(*inner_merge),
        .loop_merge = CountBarriers(*loop_merge),
    };
}

LoopBarrierLocations BarriersForUniformBreakUnderDivergentBranch() {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    Shader::IR::Block* const header = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter{*header}.Epilogue();

    Shader::IR::Block* const loop_body = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter body_ir{*loop_body};
    const Shader::IR::U32 invocation =
        body_ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U1 guard =
        body_ir.ConditionRef(body_ir.IEqual(invocation, body_ir.Imm32(0U)));
    body_ir.Epilogue();

    Shader::IR::Block* const guarded_body = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter guarded_ir{*guarded_body};
    const Shader::IR::U1 break_cond = guarded_ir.ConditionRef(guarded_ir.Imm1(true));
    guarded_ir.Epilogue();

    Shader::IR::Block* const break_skip = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter{*break_skip}.Epilogue();
    Shader::IR::Block* const inner_merge = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter{*inner_merge}.Epilogue();

    Shader::IR::Block* const continue_block = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter continue_ir{*continue_block};
    const Shader::IR::U1 repeat_cond = continue_ir.ConditionRef(continue_ir.Imm1(false));
    continue_ir.Epilogue();

    Shader::IR::Block* const loop_merge = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter{*loop_merge}.Epilogue();

    auto add_block = [&](Shader::IR::Block* block) {
        auto& node = program.syntax_list.emplace_back();
        node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
    };

    add_block(header);
    auto& loop = program.syntax_list.emplace_back();
    loop.type = Shader::IR::AbstractSyntaxNode::Type::Loop;
    loop.data.loop = {
        .body = loop_body,
        .continue_block = continue_block,
        .merge = loop_merge,
    };
    add_block(loop_body);
    auto& if_node = program.syntax_list.emplace_back();
    if_node.type = Shader::IR::AbstractSyntaxNode::Type::If;
    if_node.data.if_node = {
        .cond = guard,
        .body = guarded_body,
        .merge = inner_merge,
    };
    add_block(guarded_body);
    auto& break_node = program.syntax_list.emplace_back();
    break_node.type = Shader::IR::AbstractSyntaxNode::Type::Break;
    break_node.data.break_node = {
        .cond = break_cond,
        .merge = loop_merge,
        .skip = break_skip,
    };
    add_block(break_skip);
    auto& end_if = program.syntax_list.emplace_back();
    end_if.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
    end_if.data.end_if.merge = inner_merge;
    add_block(inner_merge);
    add_block(continue_block);
    auto& repeat = program.syntax_list.emplace_back();
    repeat.type = Shader::IR::AbstractSyntaxNode::Type::Repeat;
    repeat.data.repeat = {
        .cond = repeat_cond,
        .loop_header = header,
        .merge = loop_merge,
    };
    add_block(loop_merge);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.shared_memory_size = 256;
    runtime_info.cs_info.workgroup_size = {64, 1, 1};

    Shader::Profile profile{};
    profile.needs_lds_barriers = true;
    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);

    return {
        .inner_merge = CountBarriers(*inner_merge),
        .loop_merge = CountBarriers(*loop_merge),
    };
}

TEST(SharedMemoryBarrierPass, SynchronizesPartialGuestWaveWorkgroups) {
    EXPECT_EQ(CountInsertedBarriers(32), 1);
}

TEST(SharedMemoryBarrierPass, SynchronizesFullGuestWaveWorkgroups) {
    EXPECT_EQ(CountInsertedBarriers(64), 1);
}

TEST(SharedMemoryBarrierPass, SynchronizesMultiWaveWorkgroups) {
    EXPECT_EQ(CountInsertedBarriers(256), 1);
}

TEST(SharedMemoryBarrierPass, SkipsSynchronizationWhenHostDoesNotNeedIt) {
    EXPECT_EQ(CountInsertedBarriers(32, 256, false), 0);
}

TEST(SharedMemoryBarrierPass, SkipsWorkgroupsWithoutSharedMemory) {
    EXPECT_EQ(CountInsertedBarriers(32, 0), 0);
}

TEST(SharedMemoryBarrierPass, SkipsInvalidEmptyWorkgroups) {
    EXPECT_EQ(CountInsertedBarriers(0), 0);
}

TEST(SharedMemoryBarrierPass, SynchronizesNestedDivergenceInsideUniformGuestWave) {
    const NestedBarrierLocations barriers = BarriersForMaskedInvocationBranch({8, 8, 1});

    EXPECT_EQ(barriers.inner_merge, 1);
    EXPECT_EQ(barriers.outer_merge, 0);
}

TEST(SharedMemoryBarrierPass, KeepsBarrierOutsideNonUniformGuestWaveSelection) {
    const NestedBarrierLocations barriers = BarriersForMaskedInvocationBranch({65, 1, 1});

    EXPECT_EQ(barriers.inner_merge, 0);
    EXPECT_EQ(barriers.outer_merge, 1);
}

TEST(SharedMemoryBarrierPass, SynchronizesAtDivergentLoopMerge) {
    const LoopBarrierLocations barriers = BarriersForNestedBranchInLoop(true);

    EXPECT_EQ(barriers.inner_merge, 0);
    EXPECT_EQ(barriers.loop_merge, 1);
}

TEST(SharedMemoryBarrierPass, KeepsSynchronizationInsideUniformLoop) {
    const LoopBarrierLocations barriers = BarriersForNestedBranchInLoop(false);

    EXPECT_EQ(barriers.inner_merge, 1);
    EXPECT_EQ(barriers.loop_merge, 0);
}

TEST(SharedMemoryBarrierPass, SynchronizesAtControlDependentBreakLoopMerge) {
    const LoopBarrierLocations barriers = BarriersForUniformBreakUnderDivergentBranch();

    EXPECT_EQ(barriers.inner_merge, 0);
    EXPECT_EQ(barriers.loop_merge, 1);
}

} // namespace
