// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"

namespace {

bool HasOpcode(const Shader::IR::Block& block, Shader::IR::Opcode opcode) {
    return std::ranges::any_of(block.Instructions(), [opcode](const Shader::IR::Inst& inst) {
        return inst.GetOpcode() == opcode;
    });
}

std::vector<const Shader::IR::Inst*> ReferenceRoots(const Shader::IR::Block& block) {
    std::vector<const Shader::IR::Inst*> roots;
    for (const Shader::IR::Inst& inst : block.Instructions()) {
        if (inst.GetOpcode() == Shader::IR::Opcode::Reference) {
            roots.push_back(inst.Arg(0).Resolve().TryInstRecursive());
        }
    }
    return roots;
}

enum class SpillSourceRepresentation {
    Scalar,
    SingleLowMaskWord,
    MaskPair,
};

struct SpillRestoreResult {
    Shader::IR::Opcode opcode;
    std::optional<u32> composite_index;
};

std::optional<SpillRestoreResult> RunScalarSpillRestore(SpillSourceRepresentation representation,
                                                        Shader::IR::ScalarReg spill_source,
                                                        Shader::IR::ScalarReg restore_destination) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    Shader::Profile profile{};
    Shader::Gcn::Translator translator{program.info, runtime_info, profile};
    translator.EmitPrologue(block);

    Shader::IR::IREmitter seed_ir{*block};
    switch (representation) {
    case SpillSourceRepresentation::Scalar:
        seed_ir.SetScalarReg(Shader::IR::ScalarReg::S1, seed_ir.Imm32(1));
        seed_ir.SetScalarReg(Shader::IR::ScalarReg::S2, seed_ir.Imm32(0));
        break;
    case SpillSourceRepresentation::SingleLowMaskWord:
        seed_ir.SetThreadBitScalarReg(Shader::IR::ScalarReg::S1, seed_ir.Imm1(true));
        break;
    case SpillSourceRepresentation::MaskPair:
        seed_ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S1, seed_ir.Imm1(true));
        break;
    }

    Shader::Gcn::GcnInst write{};
    write.src[0].field = Shader::Gcn::OperandField::ScalarGPR;
    write.src[0].code = static_cast<u32>(spill_source);
    write.src[1].field = Shader::Gcn::OperandField::ConstZero;
    write.dst[0].field = Shader::Gcn::OperandField::VectorGPR;
    write.dst[0].code = 3;
    translator.V_WRITELANE_B32(write);

    Shader::Gcn::GcnInst read{};
    read.src[0].field = Shader::Gcn::OperandField::VectorGPR;
    read.src[0].code = 3;
    read.src[1].field = Shader::Gcn::OperandField::ConstZero;
    read.dst[0].field = Shader::Gcn::OperandField::ScalarGPR;
    read.dst[0].code = static_cast<u32>(restore_destination);
    translator.V_READLANE_B32(read);

    Shader::IR::IREmitter result_ir{*block};
    result_ir.Reference(result_ir.GetScalarReg(restore_destination));

    Shader::Optimization::SsaRewritePass(program);

    const auto roots = ReferenceRoots(*block);
    if (roots.size() != 1U || roots[0] == nullptr) {
        return std::nullopt;
    }
    const Shader::IR::Inst& root = *roots[0];
    const std::optional<u32> composite_index =
        root.GetOpcode() == Shader::IR::Opcode::CompositeExtractU32x4
            ? std::optional<u32>{root.Arg(1).U32()}
            : std::nullopt;
    return SpillRestoreResult{.opcode = root.GetOpcode(), .composite_index = composite_index};
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

TEST(SsaRewritePass, UnprovenMaskLaneRestoreRetainsScalarThreadView) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    const auto read_lane = ir.ReadLane(ir.Imm32(0x12345678), ir.Imm32(0));
    ir.SetScalarReg(Shader::IR::ScalarReg::S0, read_lane);
    // V_READLANE emits an optional mask restore. With no matching V_WRITELANE producer, that
    // absent shadow must not replace the ordinary scalar word's per-invocation bit view.
    ir.SetThreadBitScalarReg(Shader::IR::ScalarReg::S0,
                             ir.GetMaskLaneVariable(Shader::IR::VectorReg::V0, 0));
    ir.Reference(ir.GetThreadBitScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    const Shader::IR::Value thread_view = block->back().Arg(0).Resolve();
    ASSERT_NE(thread_view.TryInstRecursive(), nullptr);
    EXPECT_NE(thread_view.TryInstRecursive()->GetOpcode(), Shader::IR::Opcode::UndefU1);
    EXPECT_TRUE(HasOpcode(*block, Shader::IR::Opcode::LaneId));
}

TEST(SsaRewritePass, ScalarOverwriteClearsPreviousThreadMaskRepresentation) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetThreadBitScalarReg(Shader::IR::ScalarReg::S0, ir.Imm1(true));
    ir.SetScalarReg(Shader::IR::ScalarReg::S0, ir.Imm32(0));
    ir.SetScalarReg(Shader::IR::ScalarReg::S1, ir.Imm32(0));
    ir.Reference(ir.GetThreadBitScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    const Shader::IR::Value thread_view = block->back().Arg(0).Resolve();
    EXPECT_FALSE(thread_view.IsImmediate() && thread_view.U1());
    EXPECT_TRUE(HasOpcode(*block, Shader::IR::Opcode::LaneId));
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

TEST(SsaRewritePass, ThreadMaskOverwriteRetainsScalarFallback) {
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

    // The subgroup collective must remain explicit in IR so Wave64 lowering can lower or reject
    // it. A backend-only materialization would silently bypass that safety boundary.
    EXPECT_TRUE(HasOpcode(*block, Shader::IR::Opcode::Ballot));
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

    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::Ballot));
}

TEST(SsaRewritePass, ThreadMaskPairClassifiesHighWord) {
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

    const auto extract =
        std::ranges::find_if(block->Instructions(), [](const Shader::IR::Inst& inst) {
            return inst.GetOpcode() == Shader::IR::Opcode::CompositeExtractU32x4;
        });
    ASSERT_NE(extract, block->Instructions().end());
    EXPECT_EQ(extract->Arg(1).U32(), 1u);
}

TEST(SsaRewritePass, OddBaseThreadMaskPairUsesLowThenHighMaskWords) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S1, ir.Imm1(true));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S1));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S2));

    Shader::Optimization::SsaRewritePass(program);

    const auto roots = ReferenceRoots(*block);
    ASSERT_EQ(roots.size(), 2U);
    ASSERT_NE(roots[0], nullptr);
    ASSERT_NE(roots[1], nullptr);
    ASSERT_EQ(roots[0]->GetOpcode(), Shader::IR::Opcode::CompositeExtractU32x4);
    ASSERT_EQ(roots[1]->GetOpcode(), Shader::IR::Opcode::CompositeExtractU32x4);
    EXPECT_EQ(roots[0]->Arg(1).U32(), 0U);
    EXPECT_EQ(roots[1]->Arg(1).U32(), 1U);
}

TEST(SsaRewritePass, SingleOddThreadMaskWriteUsesLowMaskWord) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetThreadBitScalarReg(Shader::IR::ScalarReg::S1, ir.Imm1(true));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S1));

    Shader::Optimization::SsaRewritePass(program);

    const auto roots = ReferenceRoots(*block);
    ASSERT_EQ(roots.size(), 1U);
    ASSERT_NE(roots[0], nullptr);
    ASSERT_EQ(roots[0]->GetOpcode(), Shader::IR::Opcode::CompositeExtractU32x4);
    EXPECT_EQ(roots[0]->Arg(1).U32(), 0U);
}

TEST(SsaRewritePass, LowMaskWordRemapsWhenConsumedAsPairHighWord) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    const auto predicate = ir.IEqual(ir.LaneId(), ir.Imm32(0));
    ir.SetScalarReg(Shader::IR::ScalarReg::S0, ir.Imm32(0));
    ir.SetThreadBitScalarReg(Shader::IR::ScalarReg::S1, predicate);
    ir.Reference(ir.GetThreadBitScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    const auto extract =
        std::ranges::find_if(block->Instructions(), [](const Shader::IR::Inst& inst) {
            return inst.GetOpcode() == Shader::IR::Opcode::CompositeExtractU32x4;
        });
    ASSERT_NE(extract, block->Instructions().end());
    EXPECT_EQ(extract->Arg(1).U32(), 0U);
    EXPECT_TRUE(HasOpcode(*block, Shader::IR::Opcode::Ballot));
}

TEST(SsaRewritePass, HighMaskWordRemapsWhenConsumedAsPairLowWord) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    const auto predicate = ir.IEqual(ir.LaneId(), ir.Imm32(32));
    ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S0, predicate);
    ir.SetScalarReg(Shader::IR::ScalarReg::S2, ir.Imm32(0));
    ir.Reference(ir.GetThreadBitScalarReg(Shader::IR::ScalarReg::S1));

    Shader::Optimization::SsaRewritePass(program);

    const auto extract =
        std::ranges::find_if(block->Instructions(), [](const Shader::IR::Inst& inst) {
            return inst.GetOpcode() == Shader::IR::Opcode::CompositeExtractU32x4;
        });
    ASSERT_NE(extract, block->Instructions().end());
    EXPECT_EQ(extract->Arg(1).U32(), 1U);
    EXPECT_TRUE(HasOpcode(*block, Shader::IR::Opcode::Ballot));
}

TEST(SsaRewritePass, ThreadMaskPairReusesSinglePerInvocationValue) {
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

TEST(SsaRewritePass, SingleHighWordThreadMaskWriteUsesItsOwnRepresentation) {
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
    EXPECT_TRUE(HasOpcode(*block, Shader::IR::Opcode::LaneId));
}

TEST(SsaRewritePass, ConditionallyMaterializesMergedRepresentations) {
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

    EXPECT_TRUE(HasOpcode(*merge, Shader::IR::Opcode::Ballot));
    EXPECT_TRUE(HasOpcode(*merge, Shader::IR::Opcode::SelectU1));
}

TEST(SsaRewritePass, MergedMaskWordPositionsProduceOneWaveConsistentScalarWord) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const entry = pools.block_pool.Create(pools.inst_pool);
    auto* const high_word_path = pools.block_pool.Create(pools.inst_pool);
    auto* const scalar_path = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    entry->AddBranch(high_word_path);
    entry->AddBranch(scalar_path);
    high_word_path->AddBranch(merge);
    scalar_path->AddBranch(merge);
    program.blocks = {entry, high_word_path, scalar_path, merge};
    program.post_order_blocks = {merge, high_word_path, scalar_path, entry};

    Shader::IR::IREmitter high_ir{*high_word_path};
    high_ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S1, high_ir.Imm1(true));
    Shader::IR::IREmitter scalar_ir{*scalar_path};
    scalar_ir.SetScalarReg(Shader::IR::ScalarReg::S2, scalar_ir.Imm32(0x12345678));
    Shader::IR::IREmitter merge_ir{*merge};
    merge_ir.Reference(merge_ir.GetScalarReg(Shader::IR::ScalarReg::S2));

    Shader::Optimization::SsaRewritePass(program);

    const auto roots = ReferenceRoots(*merge);
    ASSERT_EQ(roots.size(), 1U);
    ASSERT_NE(roots[0], nullptr);
    ASSERT_EQ(roots[0]->GetOpcode(), Shader::IR::Opcode::BitwiseOr32);
    const Shader::IR::Inst* const low = roots[0]->Arg(0).TryInstRecursive();
    const Shader::IR::Inst* const high = roots[0]->Arg(1).TryInstRecursive();
    ASSERT_NE(low, nullptr);
    ASSERT_NE(high, nullptr);
    ASSERT_EQ(low->GetOpcode(), Shader::IR::Opcode::CompositeExtractU32x4);
    ASSERT_EQ(high->GetOpcode(), Shader::IR::Opcode::CompositeExtractU32x4);
    EXPECT_EQ(low->Arg(1).U32(), 0U);
    EXPECT_EQ(high->Arg(1).U32(), 1U);
    EXPECT_EQ(low->Arg(0).Resolve(), high->Arg(0).Resolve());
    EXPECT_EQ(low->Arg(0).TryInstRecursive()->GetOpcode(), Shader::IR::Opcode::Ballot);
}

TEST(SsaRewritePass, PreservesLoopBackedgeRepresentation) {
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

    EXPECT_TRUE(HasOpcode(*header, Shader::IR::Opcode::Ballot));
}

TEST(SsaRewritePass, PreservesHighMaskWordProvenanceAcrossLoopBackedge) {
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
    entry_ir.SetScalarReg(Shader::IR::ScalarReg::S2, entry_ir.Imm32(0x12345678));
    Shader::IR::IREmitter header_ir{*header};
    header_ir.Reference(header_ir.GetScalarReg(Shader::IR::ScalarReg::S2));
    Shader::IR::IREmitter body_ir{*body};
    body_ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S1, body_ir.Imm1(true));

    Shader::Optimization::SsaRewritePass(program);

    const auto roots = ReferenceRoots(*header);
    ASSERT_EQ(roots.size(), 1U);
    ASSERT_NE(roots[0], nullptr);
    EXPECT_EQ(roots[0]->GetOpcode(), Shader::IR::Opcode::BitwiseOr32);
    EXPECT_TRUE(HasOpcode(*header, Shader::IR::Opcode::Ballot));
}

TEST(SsaRewritePass, ExtractsInvocationBitFromScalarWordPair) {
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

TEST(SsaRewritePass, ProvenMaskLaneRestorePreservesThreadMaskRepresentation) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetMaskLaneVariable(Shader::IR::VectorReg::V0, 0, ir.Imm1(true));
    const auto read_lane = ir.ReadLane(ir.Imm32(0x12345678), ir.Imm32(0));
    ir.SetScalarReg(Shader::IR::ScalarReg::S0, read_lane);
    ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S0,
                                 ir.GetMaskLaneVariable(Shader::IR::VectorReg::V0, 0));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    const auto ballot =
        std::ranges::find_if(block->Instructions(), [](const Shader::IR::Inst& inst) {
            return inst.GetOpcode() == Shader::IR::Opcode::Ballot;
        });
    ASSERT_NE(ballot, block->Instructions().end());
    EXPECT_TRUE(ballot->Arg(0).Resolve().IsImmediate());
    EXPECT_TRUE(ballot->Arg(0).Resolve().U1());
}

TEST(SsaRewritePass, PartialMaskLaneProvenanceFailsClosedToScalar) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const entry = pools.block_pool.Create(pools.inst_pool);
    auto* const spill_path = pools.block_pool.Create(pools.inst_pool);
    auto* const scalar_path = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    entry->AddBranch(spill_path);
    entry->AddBranch(scalar_path);
    spill_path->AddBranch(merge);
    scalar_path->AddBranch(merge);
    program.blocks = {entry, spill_path, scalar_path, merge};
    program.post_order_blocks = {merge, spill_path, scalar_path, entry};

    Shader::IR::IREmitter spill_ir{*spill_path};
    spill_ir.SetMaskLaneVariable(Shader::IR::VectorReg::V0, 0, spill_ir.Imm1(true));
    Shader::IR::IREmitter merge_ir{*merge};
    const auto read_lane = merge_ir.ReadLane(merge_ir.Imm32(0x12345678), merge_ir.Imm32(0));
    merge_ir.SetScalarReg(Shader::IR::ScalarReg::S0, read_lane);
    merge_ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S0,
                                       merge_ir.GetMaskLaneVariable(Shader::IR::VectorReg::V0, 0));
    merge_ir.Reference(merge_ir.GetScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    EXPECT_FALSE(HasOpcode(*merge, Shader::IR::Opcode::Ballot));
    const Shader::IR::Value scalar = merge->back().Arg(0).Resolve();
    ASSERT_NE(scalar.TryInstRecursive(), nullptr);
    EXPECT_EQ(scalar.TryInstRecursive()->GetOpcode(), Shader::IR::Opcode::ReadLane);
}

TEST(SsaRewritePass, CompleteMaskLaneProvenanceSurvivesMerge) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const entry = pools.block_pool.Create(pools.inst_pool);
    auto* const left = pools.block_pool.Create(pools.inst_pool);
    auto* const right = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    entry->AddBranch(left);
    entry->AddBranch(right);
    left->AddBranch(merge);
    right->AddBranch(merge);
    program.blocks = {entry, left, right, merge};
    program.post_order_blocks = {merge, left, right, entry};

    Shader::IR::IREmitter left_ir{*left};
    left_ir.SetMaskLaneVariable(Shader::IR::VectorReg::V0, 0, left_ir.Imm1(true));
    Shader::IR::IREmitter right_ir{*right};
    right_ir.SetMaskLaneVariable(Shader::IR::VectorReg::V0, 0, right_ir.Imm1(false));
    Shader::IR::IREmitter merge_ir{*merge};
    merge_ir.SetScalarReg(Shader::IR::ScalarReg::S0, merge_ir.Imm32(0x12345678));
    merge_ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S0,
                                       merge_ir.GetMaskLaneVariable(Shader::IR::VectorReg::V0, 0));
    merge_ir.Reference(merge_ir.GetScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    EXPECT_TRUE(HasOpcode(*merge, Shader::IR::Opcode::Ballot));
}

TEST(SsaRewritePass, MatchingLaneOverwriteInvalidatesMaskLaneProvenance) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetMaskLaneVariable(Shader::IR::VectorReg::V0, 0, ir.Imm1(true));
    ir.SetVectorReg(Shader::IR::VectorReg::V0, ir.WriteLane(ir.Imm32(0), ir.Imm32(7), ir.Imm32(0)));
    ir.SetScalarReg(Shader::IR::ScalarReg::S0, ir.Imm32(0x12345678));
    ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S0,
                                 ir.GetMaskLaneVariable(Shader::IR::VectorReg::V0, 0));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::Ballot));
}

TEST(SsaRewritePass, OtherLaneOverwritePreservesMaskLaneProvenance) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetMaskLaneVariable(Shader::IR::VectorReg::V0, 0, ir.Imm1(true));
    ir.SetVectorReg(Shader::IR::VectorReg::V0, ir.WriteLane(ir.Imm32(0), ir.Imm32(7), ir.Imm32(1)));
    ir.SetScalarReg(Shader::IR::ScalarReg::S0, ir.Imm32(0x12345678));
    ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S0,
                                 ir.GetMaskLaneVariable(Shader::IR::VectorReg::V0, 0));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    EXPECT_TRUE(HasOpcode(*block, Shader::IR::Opcode::Ballot));
}

TEST(SsaRewritePass, FullVectorOverwriteInvalidatesMaskLaneProvenance) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::IR::IREmitter ir{*block};
    ir.SetMaskLaneVariable(Shader::IR::VectorReg::V0, 0, ir.Imm1(true));
    ir.SetVectorReg(Shader::IR::VectorReg::V0, ir.Imm32(7));
    ir.SetScalarReg(Shader::IR::ScalarReg::S0, ir.Imm32(0x12345678));
    ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S0,
                                 ir.GetMaskLaneVariable(Shader::IR::VectorReg::V0, 0));
    ir.Reference(ir.GetScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    EXPECT_FALSE(HasOpcode(*block, Shader::IR::Opcode::Ballot));
}

TEST(SsaRewritePass, OddBasedLowMaskWordSpillRestoresToOddBasedPair) {
    const auto result = RunScalarSpillRestore(SpillSourceRepresentation::MaskPair,
                                              Shader::IR::ScalarReg::S1, Shader::IR::ScalarReg::S5);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->opcode, Shader::IR::Opcode::CompositeExtractU32x4);
    EXPECT_EQ(result->composite_index, 0U);
}

TEST(SsaRewritePass, HighMaskWordSpillDoesNotMasqueradeAsEvenBasedLowWord) {
    const auto result = RunScalarSpillRestore(SpillSourceRepresentation::MaskPair,
                                              Shader::IR::ScalarReg::S2, Shader::IR::ScalarReg::S4);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->opcode, Shader::IR::Opcode::ReadLane);
}

TEST(SsaRewritePass, SingleOddLowMaskWordSpillRestoresAsWordZero) {
    const auto result = RunScalarSpillRestore(SpillSourceRepresentation::SingleLowMaskWord,
                                              Shader::IR::ScalarReg::S1, Shader::IR::ScalarReg::S5);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->opcode, Shader::IR::Opcode::CompositeExtractU32x4);
    EXPECT_EQ(result->composite_index, 0U);
}

TEST(SsaRewritePass, OrdinaryOddScalarSpillRetainsScalarLineage) {
    const auto result = RunScalarSpillRestore(SpillSourceRepresentation::Scalar,
                                              Shader::IR::ScalarReg::S1, Shader::IR::ScalarReg::S5);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->opcode, Shader::IR::Opcode::ReadLane);
}

TEST(SsaRewritePass, SingleDwordMaskRestorePreservesAdjacentScalarWord) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {block};
    program.post_order_blocks = {block};

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    Shader::Profile profile{};
    Shader::Gcn::Translator translator{program.info, runtime_info, profile};
    translator.EmitPrologue(block);

    Shader::IR::IREmitter seed_ir{*block};
    seed_ir.SetThreadBitScalarRegPair(Shader::IR::ScalarReg::S1, seed_ir.Imm1(true));
    seed_ir.SetScalarReg(Shader::IR::ScalarReg::S6, seed_ir.Imm32(0x12345678));

    Shader::Gcn::GcnInst write{};
    write.src[0].field = Shader::Gcn::OperandField::ScalarGPR;
    write.src[0].code = static_cast<u32>(Shader::IR::ScalarReg::S1);
    write.src[1].field = Shader::Gcn::OperandField::ConstZero;
    write.dst[0].field = Shader::Gcn::OperandField::VectorGPR;
    write.dst[0].code = 3;
    translator.V_WRITELANE_B32(write);

    Shader::Gcn::GcnInst read{};
    read.src[0].field = Shader::Gcn::OperandField::VectorGPR;
    read.src[0].code = 3;
    read.src[1].field = Shader::Gcn::OperandField::ConstZero;
    read.dst[0].field = Shader::Gcn::OperandField::ScalarGPR;
    read.dst[0].code = static_cast<u32>(Shader::IR::ScalarReg::S5);
    translator.V_READLANE_B32(read);

    Shader::IR::IREmitter result_ir{*block};
    result_ir.Reference(result_ir.GetScalarReg(Shader::IR::ScalarReg::S6));

    Shader::Optimization::SsaRewritePass(program);

    const Shader::IR::Value untouched_word = block->back().Arg(0).Resolve();
    ASSERT_TRUE(untouched_word.IsImmediate());
    EXPECT_EQ(untouched_word.U32(), 0x12345678U);
}

TEST(SsaRewritePass, LastSgprCanRetainSingleDwordMaskRestore) {
    const auto result =
        RunScalarSpillRestore(SpillSourceRepresentation::MaskPair, Shader::IR::ScalarReg::S1,
                              Shader::IR::ScalarReg::S103);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->opcode, Shader::IR::Opcode::CompositeExtractU32x4);
    EXPECT_EQ(result->composite_index, 0U);
}
