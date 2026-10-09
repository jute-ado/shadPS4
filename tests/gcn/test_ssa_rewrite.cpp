// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/passes/sharp_source_traversal.h"
#include "shader_recompiler/recompiler.h"

namespace {

bool HasOpcode(const Shader::IR::Block& block, Shader::IR::Opcode opcode) {
    return std::ranges::any_of(block.Instructions(), [opcode](const Shader::IR::Inst& inst) {
        return inst.GetOpcode() == opcode;
    });
}

} // namespace

TEST(SsaRewritePass, ReusesUndefinitionsAcrossDisconnectedBlocks) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const first = pools.block_pool.Create(pools.inst_pool);
    auto* const second = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {first, second};
    program.post_order_blocks = {second, first};

    Shader::IR::IREmitter first_ir{*first};
    first_ir.Reference(first_ir.GetScalarReg(Shader::IR::ScalarReg::S0));
    Shader::IR::IREmitter second_ir{*second};
    second_ir.Reference(second_ir.GetScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    const Shader::IR::Value first_undef = first->back().Arg(0).Resolve();
    const Shader::IR::Value second_undef = second->back().Arg(0).Resolve();
    ASSERT_NE(first_undef.TryInstRecursive(), nullptr);
    EXPECT_EQ(first_undef.TryInstRecursive()->GetOpcode(), Shader::IR::Opcode::UndefU32);
    EXPECT_EQ(first_undef, second_undef);
    EXPECT_EQ(first_undef.TryInstRecursive()->GetParent(), first);
}

TEST(SsaRewritePass, MaterializesThreadBitScalarWhenReadAsScalarWord) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetThreadBitScalarReg(Shader::IR::ScalarReg::S0, ir.Imm1(true));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    EXPECT_TRUE(HasOpcode(*block, Shader::IR::Opcode::Ballot));
    EXPECT_TRUE(HasOpcode(*block, Shader::IR::Opcode::CompositeExtractU32x4));
}

TEST(SsaRewritePass, ThreadMaskOverwriteRetainsScalarFallbackForResourceTracking) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetScalarReg(Shader::IR::ScalarReg::S0, ir.Imm32(0x12345678));
    ir.SetThreadBitScalarReg(Shader::IR::ScalarReg::S0, ir.Imm1(true));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    EXPECT_TRUE(HasOpcode(*block, Shader::IR::Opcode::MaterializeScalarMaskWord));
    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::Ballot));
}

TEST(SsaRewritePass, SingleWordThreadMaskWritePreservesAdjacentScalarWord) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetScalarReg(Shader::IR::ScalarReg::S1, ir.Imm32(0x12345678));
    ir.SetThreadBitScalarReg(Shader::IR::ScalarReg::S0, ir.Imm1(true));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S1));

    Shader::Optimization::SsaRewritePass(program);

    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::MaterializeScalarMaskWord));
    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::Ballot));
}

TEST(SsaRewritePass, ThreadMaskWriteClassifiesHighWordAsSameMask) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S0, ir.Imm1(true));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S1));

    Shader::Optimization::SsaRewritePass(program);

    const auto extract = std::ranges::find_if(
        block->Instructions(), [](const Shader::IR::Inst& inst) {
            return inst.GetOpcode() == Shader::IR::Opcode::CompositeExtractU32x4;
        });
    ASSERT_NE(extract, block->Instructions().end());
    EXPECT_EQ(extract->Arg(1).U32(), 1u);
}

TEST(SsaRewritePass, ThreadMaskPairUsesSinglePerInvocationValue) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S0, ir.Imm1(false));
    ir.Reference(ir.GetThreadBitScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::LaneId));
    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::SelectU1));
}

TEST(SsaRewritePass, SingleHighWordThreadMaskWriteUsesItsOwnInvocationValue) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetScalarReg(Shader::IR::ScalarReg::S0, ir.Imm32(0));
    ir.SetThreadBitScalarReg(Shader::IR::ScalarReg::S1, ir.Imm1(true));
    ir.Reference(ir.GetThreadBitScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::UndefU1));
}

TEST(SsaRewritePass, ConditionallyMaterializesMergedThreadBitAndScalarWord) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const entry = pools.block_pool.Create(pools.inst_pool);
    auto* const thread_mask_path = pools.block_pool.Create(pools.inst_pool);
    auto* const scalar_path = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    entry->AddBranch(thread_mask_path);
    entry->AddBranch(scalar_path);
    thread_mask_path->AddBranch(merge);
    scalar_path->AddBranch(merge);
    program.blocks = {entry, thread_mask_path, scalar_path, merge};
    program.post_order_blocks = {merge, thread_mask_path, scalar_path, entry};

    Shader::IR::IREmitter mask_ir{*thread_mask_path};
    mask_ir.SetThreadBitScalarReg(Shader::IR::ScalarReg::S0, mask_ir.Imm1(true));
    Shader::IR::IREmitter scalar_ir{*scalar_path};
    scalar_ir.SetScalarReg(Shader::IR::ScalarReg::S0, scalar_ir.Imm32(0x12345678));
    Shader::IR::IREmitter merge_ir{*merge};
    merge_ir.Reference(merge_ir.GetScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    EXPECT_TRUE(HasOpcode(*merge, Shader::IR::Opcode::MaterializeScalarMaskWord));
}

TEST(SsaRewritePass, PreservesLoopBackedgeMaskRepresentation) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const entry = pools.block_pool.Create(pools.inst_pool);
    auto* const header = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const exit = pools.block_pool.Create(pools.inst_pool);
    entry->AddBranch(header);
    header->AddBranch(body);
    header->AddBranch(exit);
    body->AddBranch(header);
    program.blocks = {entry, header, body, exit};
    program.post_order_blocks = {exit, body, header, entry};

    Shader::IR::IREmitter entry_ir{*entry};
    entry_ir.SetScalarReg(Shader::IR::ScalarReg::S0, entry_ir.Imm32(0x12345678));
    Shader::IR::IREmitter header_ir{*header};
    header_ir.Reference(header_ir.GetScalarReg(Shader::IR::ScalarReg::S0));
    Shader::IR::IREmitter body_ir{*body};
    body_ir.SetThreadBitScalarReg(Shader::IR::ScalarReg::S0, body_ir.Imm1(true));

    Shader::Optimization::SsaRewritePass(program);

    EXPECT_TRUE(HasOpcode(*header, Shader::IR::Opcode::MaterializeScalarMaskWord));
}

TEST(SsaRewritePass, ExtractsInvocationBitWhenScalarWordsAreReadAsThreadMask) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetScalarReg(Shader::IR::ScalarReg::S0, ir.Imm32(1));
    ir.SetScalarReg(Shader::IR::ScalarReg::S1, ir.Imm32(0));
    ir.Reference(ir.GetThreadBitScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    EXPECT_TRUE(HasOpcode(*block, Shader::IR::Opcode::LaneId));
    EXPECT_TRUE(HasOpcode(*block, Shader::IR::Opcode::ShiftRightLogical32));
    EXPECT_TRUE(HasOpcode(*block, Shader::IR::Opcode::INotEqual32));
    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::UndefU1));
}

TEST(SharpSourceTraversal, ConditionalMaskConversionTraversesOnlyScalarFallback) {
    constexpr auto range = Shader::Optimization::SharpSourceTraversalRange(
        Shader::IR::Opcode::MaterializeScalarMaskWord, 4);

    EXPECT_EQ(range.begin, 2u);
    EXPECT_EQ(range.end, 3u);
}
