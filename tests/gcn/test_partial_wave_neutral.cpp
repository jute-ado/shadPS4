// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <limits>
#include <gtest/gtest.h>

#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"

namespace {
using namespace Shader;

struct PartialWaveFixture {
    explicit PartialWaveFixture(std::array<u32, 3> shape = {32U, 1U, 1U}) : program{info} {
        info.stage = Stage::Compute;
        info.l_stage = LogicalStage::Compute;
        runtime.Initialize(Stage::Compute);
        runtime.cs_info.workgroup_size = shape;
        profile.subgroup_size = 32;
        entry = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(entry);
        auto& node = program.syntax_list.emplace_back();
        node.type = IR::AbstractSyntaxNode::Type::Block;
        node.data.block = entry;
        program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
        Gcn::Translator translator{info, runtime, profile};
        translator.EmitPrologue(entry);
        const auto initial_set = std::ranges::find_if(
            *entry, [](const IR::Inst& inst) { return inst.GetOpcode() == IR::Opcode::SetExec; });
        EXPECT_NE(initial_set, entry->end());
        initial_mask = IR::U1{initial_set->Arg(0)};
    }

    void Run() {
        Optimization::LowerLaneIdToLocalInvocationIndexPass(program, runtime, profile);
    }

    Info info{};
    IR::Program program;
    Pools pools{};
    RuntimeInfo runtime{};
    Profile profile{};
    IR::Block* entry{};
    IR::U1 initial_mask;
};

enum class NeutralOperation { Minimum, Add };

IR::U32 EmitReduction(IR::IREmitter& ir, const IR::U1& mask, NeutralOperation operation,
                      u32 first_peer = 16U, bool neutral_is_undefined = false) {
    const auto neutral = operation == NeutralOperation::Minimum
                             ? ir.Imm32(std::numeric_limits<float>::infinity())
                             : ir.Imm32(0.0f);
    const auto active = ir.BitCast<IR::F32>(ir.GetUserData(IR::ScalarReg::S0));
    IR::F32 inactive = neutral;
    if (neutral_is_undefined) {
        ir.block->AppendNewInst(IR::Opcode::UndefU32, {});
        inactive = ir.BitCast<IR::F32>(IR::U32{IR::Value{&ir.block->back()}});
    }
    IR::F32 current{ir.Select(mask, active, inactive)};
    for (const u32 distance : {first_peer, 8U, 4U, 2U, 1U}) {
        const IR::U32 lane = ir.BitwiseOr(ir.BitwiseAnd(ir.LaneId(), ir.Imm32(255)), ir.Imm32(0));
        const auto peer = ir.BitwiseXor(lane, ir.Imm32(distance));
        const auto transferred =
            ir.BitCast<IR::F32>(ir.ReadLane(ir.BitCast<IR::U32>(current), peer));
        current = operation == NeutralOperation::Minimum ? IR::F32{ir.FPMin(current, transferred)}
                                                         : IR::F32{ir.FPAdd(current, transferred)};
    }
    return ir.BitCast<IR::U32>(current);
}

TEST(PartialWaveNeutralRead, ExplicitlyInitializedInactiveReductionHasItsExactNeutralBits) {
    for (const auto operation : {NeutralOperation::Minimum, NeutralOperation::Add}) {
        for (const auto shape : {std::array{32U, 1U, 1U}, std::array{4U, 4U, 2U}}) {
            PartialWaveFixture fixture{shape};
            IR::IREmitter ir{*fixture.entry};
            ir.SetExec(ir.LogicalOr(ir.LogicalNot(fixture.initial_mask), fixture.initial_mask));
            const auto value = EmitReduction(ir, fixture.initial_mask, operation);
            // V_READLANE ignores EXEC: restoring the partial mask after the value is fully
            // initialized must not erase that value's inactive-lane provenance.
            ir.SetExec(fixture.initial_mask);
            const auto upper = ir.ReadLane(value, ir.Imm32(32));
            ir.Reference(upper);
            auto* const reference = &fixture.entry->back();
            fixture.Run();
            ASSERT_TRUE(reference->Arg(0).IsImmediate());
            EXPECT_EQ(reference->Arg(0).U32(),
                      operation == NeutralOperation::Minimum ? 0x7f800000U : 0U);
            EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 0U);
            EXPECT_EQ(std::ranges::count_if(*fixture.entry,
                                            [](const IR::Inst& inst) {
                                                return inst.GetOpcode() == IR::Opcode::Barrier;
                                            }),
                      0);
        }
    }
}

TEST(PartialWaveNeutralRead, AnArbitraryInactiveRegisterReadMustNotBecomeZero) {
    PartialWaveFixture fixture;
    IR::IREmitter ir{*fixture.entry};
    ir.SetExec(ir.Imm1(true));
    const auto upper = ir.ReadLane(ir.GetUserData(IR::ScalarReg::S0), ir.Imm32(32));
    ir.Reference(upper);
    auto* const reference = &fixture.entry->back();
    fixture.Run();
    ASSERT_FALSE(reference->Arg(0).IsImmediate());
    EXPECT_EQ(reference->Arg(0).Inst()->GetOpcode(), IR::Opcode::ReadLane);
}

TEST(PartialWaveNeutralRead, SavedMaskSelfPhiRequiresAnEqualDefinedExternalSeed) {
    for (const bool conflicting_backedge : {false, true}) {
        PartialWaveFixture fixture;
        auto* const header = fixture.pools.block_pool.Create(fixture.pools.inst_pool);
        auto* const body = fixture.pools.block_pool.Create(fixture.pools.inst_pool);
        auto* const exit = fixture.pools.block_pool.Create(fixture.pools.inst_pool);
        fixture.entry->AddBranch(header);
        header->AddBranch(body);
        body->AddBranch(header);
        body->AddBranch(exit);
        fixture.program.blocks.insert(fixture.program.blocks.end(), {header, body, exit});
        fixture.program.syntax_list.pop_back();
        auto append_block = [&](IR::Block* block) {
            auto& node = fixture.program.syntax_list.emplace_back();
            node.type = IR::AbstractSyntaxNode::Type::Block;
            node.data.block = block;
        };
        append_block(header);
        auto& loop = fixture.program.syntax_list.emplace_back();
        loop.type = IR::AbstractSyntaxNode::Type::Loop;
        loop.data.loop.body = body;
        loop.data.loop.continue_block = body;
        loop.data.loop.merge = exit;
        append_block(body);
        auto& repeat = fixture.program.syntax_list.emplace_back();
        repeat.type = IR::AbstractSyntaxNode::Type::Repeat;
        repeat.data.repeat.cond = IR::U1{IR::Value{false}};
        repeat.data.repeat.loop_header = header;
        repeat.data.repeat.merge = exit;
        append_block(exit);
        fixture.program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;

        auto* const phi = &*header->PrependNewInst(header->begin(), IR::Opcode::Phi);
        phi->SetFlags(IR::Type::U1);
        phi->AddPhiOperand(fixture.entry, fixture.initial_mask);
        phi->AddPhiOperand(body, conflicting_backedge ? IR::Value{true} : IR::Value{phi});
        IR::IREmitter ir{*body};
        const IR::U1 saved_mask{IR::Value{phi}};
        ir.SetExec(ir.LogicalOr(saved_mask, ir.LogicalNot(saved_mask)));
        const auto value = EmitReduction(ir, saved_mask, NeutralOperation::Minimum);
        ir.Reference(ir.ReadLane(value, ir.Imm32(32)));
        auto* const reference = &body->back();
        fixture.Run();
        if (conflicting_backedge) {
            EXPECT_FALSE(reference->Arg(0).IsImmediate());
        } else {
            ASSERT_TRUE(reference->Arg(0).IsImmediate());
            EXPECT_EQ(reference->Arg(0).U32(), 0x7f800000U);
        }
    }
}

TEST(PartialWaveNeutralRead, UnknownOrInitiallyActiveMasksCannotProveInactiveInitialization) {
    for (const bool literal_true : {false, true}) {
        PartialWaveFixture fixture;
        IR::IREmitter ir{*fixture.entry};
        ir.SetExec(ir.Imm1(true));
        const IR::U1 mask = literal_true
                                ? ir.Imm1(true)
                                : ir.IEqual(ir.GetUserData(IR::ScalarReg::S1), ir.Imm32(0));
        const auto value = EmitReduction(ir, mask, NeutralOperation::Add);
        ir.Reference(ir.ReadLane(value, ir.Imm32(32)));
        auto* const reference = &fixture.entry->back();
        fixture.Run();
        EXPECT_FALSE(reference->Arg(0).IsImmediate());
    }
}

TEST(PartialWaveNeutralRead, InactiveValueProducersNeedAnExplicitFullExecInterval) {
    for (const bool late_restore : {false, true}) {
        PartialWaveFixture fixture;
        IR::IREmitter ir{*fixture.entry};
        const auto value = EmitReduction(ir, fixture.initial_mask, NeutralOperation::Add);
        if (late_restore) {
            ir.SetExec(ir.Imm1(true));
        }
        ir.Reference(ir.ReadLane(value, ir.Imm32(32)));
        auto* const reference = &fixture.entry->back();
        fixture.Run();
        EXPECT_FALSE(reference->Arg(0).IsImmediate());
    }
}

TEST(PartialWaveNeutralRead, APostRestoreBitcastPreservesButArithmeticCannotDefineInactiveBits) {
    for (const bool late_arithmetic : {false, true}) {
        PartialWaveFixture fixture;
        IR::IREmitter ir{*fixture.entry};
        ir.SetExec(ir.Imm1(true));
        const IR::F32 initialized{ir.Select(fixture.initial_mask,
                                            ir.BitCast<IR::F32>(ir.GetUserData(IR::ScalarReg::S0)),
                                            ir.Imm32(0.0f))};
        ir.SetExec(fixture.initial_mask);
        const IR::F32 source =
            late_arithmetic ? IR::F32{ir.FPAdd(initialized, initialized)} : initialized;
        ir.Reference(ir.ReadLane(ir.BitCast<IR::U32>(source), ir.Imm32(32)));
        auto* const reference = &fixture.entry->back();
        fixture.Run();
        if (late_arithmetic) {
            EXPECT_FALSE(reference->Arg(0).IsImmediate());
        } else {
            ASSERT_TRUE(reference->Arg(0).IsImmediate());
            EXPECT_EQ(reference->Arg(0).U32(), 0U);
        }
    }
}

TEST(PartialWaveNeutralRead, APeerThatEntersTheActiveHalfCannotInventTheActiveValue) {
    PartialWaveFixture fixture;
    IR::IREmitter ir{*fixture.entry};
    ir.SetExec(ir.Imm1(true));
    const auto value = EmitReduction(ir, fixture.initial_mask, NeutralOperation::Add, 32U);
    ir.Reference(ir.ReadLane(value, ir.Imm32(32)));
    auto* const reference = &fixture.entry->back();
    fixture.Run();
    EXPECT_FALSE(reference->Arg(0).IsImmediate());
}

TEST(PartialWaveNeutralRead, UndefinedNeutralAndFullWaveRemainConservative) {
    for (const bool full_wave : {false, true}) {
        PartialWaveFixture fixture{{full_wave ? 64U : 32U, 1U, 1U}};
        fixture.profile.max_shared_memory_size = 0;
        IR::IREmitter ir{*fixture.entry};
        ir.SetExec(ir.Imm1(true));
        const auto value =
            EmitReduction(ir, fixture.initial_mask, NeutralOperation::Add, 16U, !full_wave);
        ir.Reference(ir.ReadLane(value, ir.Imm32(32)));
        auto* const reference = &fixture.entry->back();
        fixture.Run();
        EXPECT_FALSE(reference->Arg(0).IsImmediate());
    }
}

TEST(PartialWaveNeutralRead, MinimumRecoveryDoesNotInventDenormalOrNonIdentitySemantics) {
    for (const u32 bits : {1U, 0U, 0x3f800000U, 0xff800000U, 0x7fc00000U}) {
        PartialWaveFixture fixture;
        IR::IREmitter ir{*fixture.entry};
        ir.SetExec(ir.Imm1(true));
        const auto selected = IR::F32{
            ir.Select(fixture.initial_mask, ir.BitCast<IR::F32>(ir.GetUserData(IR::ScalarReg::S0)),
                      ir.BitCast<IR::F32>(ir.Imm32(bits)))};
        const IR::F32 reduced{ir.FPMin(selected, selected)};
        ir.Reference(ir.ReadLane(ir.BitCast<IR::U32>(reduced), ir.Imm32(32)));
        auto* const reference = &fixture.entry->back();
        fixture.Run();
        EXPECT_FALSE(reference->Arg(0).IsImmediate()) << "bits=" << bits;
    }
}

TEST(PartialWaveNeutralRead, InitialPredicateProofHasABoundedInstructionBudget) {
    PartialWaveFixture fixture;
    const auto initial_set = std::ranges::find_if(*fixture.entry, [](const IR::Inst& inst) {
        return inst.GetOpcode() == IR::Opcode::SetExec;
    });
    IR::IREmitter prefix{*fixture.entry, initial_set};
    const auto zero_tree = [&](auto&& self, u32 depth) -> IR::U32 {
        return depth == 0U ? prefix.Imm32(0)
                           : IR::U32{prefix.IAdd(self(self, depth - 1U), self(self, depth - 1U))};
    };
    const IR::U32 inflated_linear =
        prefix.IAdd(IR::U32{fixture.initial_mask.Inst()->Arg(0)}, zero_tree(zero_tree, 9U));
    fixture.initial_mask = prefix.ILessThan(inflated_linear, prefix.Imm32(32), false);
    initial_set->SetArg(0, fixture.initial_mask);
    IR::IREmitter ir{*fixture.entry};
    ir.SetExec(ir.Imm1(true));
    const auto value = EmitReduction(ir, fixture.initial_mask, NeutralOperation::Add);
    ir.Reference(ir.ReadLane(value, ir.Imm32(32)));
    auto* const reference = &fixture.entry->back();
    fixture.Run();
    EXPECT_FALSE(reference->Arg(0).IsImmediate());
}

} // namespace
