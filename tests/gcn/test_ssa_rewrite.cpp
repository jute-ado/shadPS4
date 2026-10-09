// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/passes/sharp_source_traversal.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

namespace {

bool HasOpcode(const Shader::IR::Block& block, Shader::IR::Opcode opcode) {
    return std::ranges::any_of(block.Instructions(), [opcode](const Shader::IR::Inst& inst) {
        return inst.GetOpcode() == opcode;
    });
}

size_t CountOpcode(const Shader::IR::Block& block, Shader::IR::Opcode opcode) {
    return std::ranges::count_if(block.Instructions(), [opcode](const Shader::IR::Inst& inst) {
        return inst.GetOpcode() == opcode;
    });
}

} // namespace

TEST(SsaDestroyPass, RemovesForwardIdentityLeftByPhiReplacement) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const first = pools.block_pool.Create(pools.inst_pool);
    auto* const second = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {first, second, merge};
    program.post_order_blocks = {merge, second, first};

    Shader::IR::Inst* const phi =
        &*merge->PrependNewInst(merge->begin(), Shader::IR::Opcode::Phi);
    phi->SetFlags(Shader::IR::Type::U32);
    phi->AddPhiOperand(first, Shader::IR::Value{1U});
    phi->AddPhiOperand(second, Shader::IR::Value{2U});
    Shader::IR::IREmitter merge_ir{*merge};
    merge_ir.Reference(Shader::IR::U32{phi});

    Shader::Optimization::SsaDestroyPass(program);

    EXPECT_FALSE(HasOpcode(*merge, Shader::IR::Opcode::Identity));
    EXPECT_TRUE(HasOpcode(*merge, Shader::IR::Opcode::GetVirtualRegister));
    EXPECT_TRUE(HasOpcode(*first, Shader::IR::Opcode::SetVirtualRegister));
    EXPECT_TRUE(HasOpcode(*second, Shader::IR::Opcode::SetVirtualRegister));
}

TEST(SsaDestroyPass, SpillsCrossBlockValuesForSyntheticDispatcherDominance) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const producer = pools.block_pool.Create(pools.inst_pool);
    auto* const consumer = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {producer, consumer};
    program.post_order_blocks = {consumer, producer};

    Shader::IR::IREmitter producer_ir{*producer};
    const Shader::IR::U32 value = producer_ir.IAdd(producer_ir.Imm32(1), producer_ir.Imm32(2));
    Shader::IR::IREmitter consumer_ir{*consumer};
    consumer_ir.Reference(value);

    Shader::Optimization::SsaDestroyPass(program, true);

    EXPECT_TRUE(HasOpcode(*producer, Shader::IR::Opcode::SetVirtualRegister));
    EXPECT_TRUE(HasOpcode(*consumer, Shader::IR::Opcode::GetVirtualRegister));
    EXPECT_EQ(consumer->back().Arg(0).TryInstRecursive()->GetParent(), consumer);
}

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

TEST(SsaRewritePass, GuestExecPredicatesVectorRegisterWritesForLockstepControl) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetExec(ir.Imm1(true));
    ir.SetVectorReg(Shader::IR::VectorReg::V0, ir.Imm32(0x11111111));
    ir.SetExec(ir.Imm1(false));
    ir.SetVectorReg(Shader::IR::VectorReg::V0, ir.Imm32(0x22222222));
    ir.Reference(ir.GetVectorReg(Shader::IR::VectorReg::V0));

    Shader::Optimization::SsaRewritePass(program, true);

    EXPECT_TRUE(HasOpcode(*block, Shader::IR::Opcode::SelectU32));
}

TEST(SsaRewritePass, GuestExecDoesNotPredicateWriteLaneRegisterTransfer) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetExec(ir.Imm1(true));
    ir.SetVectorReg(Shader::IR::VectorReg::V0, ir.Imm32(0x11111111));
    ir.SetExec(ir.Imm1(false));
    const Shader::IR::U32 lane_write =
        ir.WriteLane(ir.GetVectorReg(Shader::IR::VectorReg::V0), ir.Imm32(0x22222222),
                     ir.Imm32(7));
    ir.SetVectorReg(Shader::IR::VectorReg::V0, lane_write);
    ir.Reference(ir.GetVectorReg(Shader::IR::VectorReg::V0));

    Shader::Optimization::SsaRewritePass(program, true);

    const Shader::IR::Inst* const result = block->back().Arg(0).Resolve().TryInstRecursive();
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetOpcode(), Shader::IR::Opcode::WriteLane);
}

TEST(ConstantPropagationPass, LowersDynamicFloatClassMaskWithoutAssertion) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.Reference(ir.FPCmpClass32(
        ir.GetAttribute(Shader::IR::Attribute::FragCoord, 0),
        ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0)));

    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);

    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::FPCmpClass32));
    EXPECT_TRUE(HasOpcode(*block, Shader::IR::Opcode::BitwiseAnd32));
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

TEST(SsaRewritePass, ThreadMaskPairDoesNotCaptureStaleScalarWords) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetScalarReg(Shader::IR::ScalarReg::S0, ir.Imm32(0x11111111));
    ir.SetScalarReg(Shader::IR::ScalarReg::S1, ir.Imm32(0x22222222));
    ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S0, ir.Imm1(true));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S0));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S1));

    Shader::Optimization::SsaRewritePass(program);

    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::CaptureScalarMaskWord));
}

TEST(SsaRewritePass, SplitWave64ThreadMaskPairCapturesBothExactScalarWords) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetScalarReg(Shader::IR::ScalarReg::S0, ir.Imm32(0x11111111));
    ir.SetScalarReg(Shader::IR::ScalarReg::S1, ir.Imm32(0x22222222));
    ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S0, ir.Imm1(true));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S0));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S1));

    Shader::Optimization::SsaRewritePass(program, false, true);

    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::CaptureScalarMaskWord), 2U);
    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::Ballot));
}

TEST(SsaRewritePass, ScalarPairCopyPreservesMaskRepresentationWithoutMaterializing) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U1 mask = ir.Imm1(true);
    ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S0, mask);
    ir.CopyScalarRegPair(Shader::IR::ScalarReg::S2, Shader::IR::ScalarReg::S0);
    ir.Reference(ir.GetThreadBitScalarReg(Shader::IR::ScalarReg::S2));

    Shader::Optimization::SsaRewritePass(program);

    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::MaterializeScalarMaskWord));
    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::Ballot));
    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::LaneId));
}

TEST(SsaRewritePass, OverlappingScalarPairCopySnapshotsBothSourceWords) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetScalarReg(Shader::IR::ScalarReg::S0, ir.Imm32(0x11111111));
    ir.SetScalarReg(Shader::IR::ScalarReg::S1, ir.Imm32(0x22222222));
    ir.CopyScalarRegPair(Shader::IR::ScalarReg::S1, Shader::IR::ScalarReg::S0);
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S1));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S2));

    Shader::Optimization::SsaRewritePass(program);

    auto reference = std::ranges::find_if(block->Instructions(), [](const Shader::IR::Inst& inst) {
        return inst.GetOpcode() == Shader::IR::Opcode::Reference;
    });
    ASSERT_NE(reference, block->Instructions().end());
    EXPECT_EQ(reference->Arg(0).Resolve().U32(), 0x11111111u);
    reference = std::ranges::find_if(++reference, block->Instructions().end(),
                                     [](const Shader::IR::Inst& inst) {
                                         return inst.GetOpcode() == Shader::IR::Opcode::Reference;
                                     });
    ASSERT_NE(reference, block->Instructions().end());
    EXPECT_EQ(reference->Arg(0).Resolve().U32(), 0x22222222u);
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

TEST(SsaRewritePass, PreservesHostSubgroupPathForScalarOnlyMaskPhiCycle) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const entry = pools.block_pool.Create(pools.inst_pool);
    auto* const header = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const scalar_write = pools.block_pool.Create(pools.inst_pool);
    auto* const skip_write = pools.block_pool.Create(pools.inst_pool);
    auto* const latch = pools.block_pool.Create(pools.inst_pool);
    auto* const exit = pools.block_pool.Create(pools.inst_pool);
    entry->AddBranch(header);
    header->AddBranch(body);
    header->AddBranch(exit);
    body->AddBranch(scalar_write);
    body->AddBranch(skip_write);
    scalar_write->AddBranch(latch);
    skip_write->AddBranch(latch);
    latch->AddBranch(header);
    program.blocks = {entry, header, body, scalar_write, skip_write, latch, exit};
    program.post_order_blocks = {exit, latch, scalar_write, skip_write, body, header, entry};

    Shader::IR::IREmitter entry_ir{*entry};
    entry_ir.SetScalarReg(Shader::IR::ScalarReg::S0, entry_ir.Imm32(7));
    Shader::IR::IREmitter header_ir{*header};
    header_ir.Reference(header_ir.LaneId());
    header_ir.Reference(header_ir.GetScalarReg(Shader::IR::ScalarReg::S0));
    Shader::IR::IREmitter write_ir{*scalar_write};
    write_ir.SetScalarReg(Shader::IR::ScalarReg::S0, write_ir.Imm32(9));

    Shader::Optimization::SsaRewritePass(program);

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    Shader::Profile profile{};
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 64 * 1024;
    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile);

    EXPECT_FALSE(HasOpcode(*header, Shader::IR::Opcode::MaterializeScalarMaskWord));
    EXPECT_TRUE(HasOpcode(*header, Shader::IR::Opcode::LaneId));
    EXPECT_TRUE(program.info.requires_subgroup_uniform_control_flow);
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

TEST(SharpSourceTraversal, Wave64MaskExchangeTraversesOnlyScalarFallback) {
    constexpr auto range = Shader::Optimization::SharpSourceTraversalRange(
        Shader::IR::Opcode::Wave64MaterializeScalarMaskWord, 6);
    EXPECT_EQ(range.begin, 2U);
    EXPECT_EQ(range.end, 3U);
}

TEST(SharpSourceTraversal, Wave64LockstepMaskExchangeTraversesOnlyScalarFallback) {
    constexpr auto range = Shader::Optimization::SharpSourceTraversalRange(
        Shader::IR::Opcode::Wave64LockstepMaterializeScalarMaskWord, 6);
    EXPECT_EQ(range.begin, 2U);
    EXPECT_EQ(range.end, 3U);
}

TEST(SharpSourceTraversal, ReadLaneTraversesOnlyTheCarriedValue) {
    const auto range = Shader::Optimization::SharpSourceTraversalRange(
        Shader::IR::Opcode::ReadLane, 2);
    EXPECT_EQ(range.begin, 0U);
    EXPECT_EQ(range.end, 1U);
}

TEST(SharpSourceTraversal, ReadFirstLaneTraversesOnlyTheCarriedValue) {
    const auto range = Shader::Optimization::SharpSourceTraversalRange(
        Shader::IR::Opcode::ReadFirstLane, 2);
    EXPECT_EQ(range.begin, 0U);
    EXPECT_EQ(range.end, 1U);

    const auto wave64_range = Shader::Optimization::SharpSourceTraversalRange(
        Shader::IR::Opcode::Wave64ReadFirstActiveLane, 4);
    EXPECT_EQ(wave64_range.begin, 0U);
    EXPECT_EQ(wave64_range.end, 1U);
}

TEST(SharpSourceTraversal, WriteLaneTraversesOldAndReplacementValues) {
    const auto range = Shader::Optimization::SharpSourceTraversalRange(
        Shader::IR::Opcode::WriteLane, 3);
    EXPECT_EQ(range.begin, 0U);
    EXPECT_EQ(range.end, 2U);
}
