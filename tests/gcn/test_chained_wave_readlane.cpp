// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <gtest/gtest.h>

#include "gcn_test_runner.hpp"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/runtime_info.h"
#include "spirv/unified1/spirv.hpp"

namespace {
using namespace Shader;

struct ChainedWaveFixture {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    RuntimeInfo runtime{};
    Profile profile{};
    std::array<IR::Block*, 5> blocks{};
    IR::Inst* selected{};

    explicit ChainedWaveFixture(IR::Opcode operation = IR::Opcode::ReadLane,
                                bool undefined_source = false, bool restore_first = false) {
        info.stage = Stage::Compute;
        info.l_stage = LogicalStage::Compute;
        runtime.Initialize(Stage::Compute);
        runtime.cs_info.workgroup_size = {64, 1, 1};
        profile.subgroup_size = 32;
        profile.max_shared_memory_size = 32 * 1024;
        profile.supported_spirv = 0x00010600;
        for (auto& block : blocks) {
            block = pools.block_pool.Create(pools.inst_pool);
            program.blocks.push_back(block);
        }
        blocks[0]->AddBranch(blocks[1]);
        blocks[0]->AddBranch(blocks[2]);
        blocks[1]->AddBranch(blocks[2]);
        blocks[2]->AddBranch(blocks[3]);
        blocks[2]->AddBranch(blocks[4]);
        blocks[3]->AddBranch(blocks[4]);

        IR::IREmitter entry{*blocks[0]};
        entry.SetExec(entry.Imm1(true));
        const auto invocation = entry.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
        const auto first = entry.ConditionRef(entry.ILessThan(invocation, entry.Imm32(16U), false));
        entry.SetExec(first);
        IR::IREmitter interior{*blocks[1]};
        interior.Reference(interior.IAdd(invocation, interior.Imm32(1U)));
        IR::IREmitter atomic_header{*blocks[2]};
        atomic_header.SetExec(atomic_header.Imm1(true));
        const auto second =
            atomic_header.ConditionRef(atomic_header.IEqual(invocation, atomic_header.Imm32(0U)));
        atomic_header.SetExec(second);
        IR::IREmitter atomic_body{*blocks[3]};
        const IR::U32 atomic{atomic_body.BufferAtomicIAdd(
            atomic_body.Imm32(0U), atomic_body.Imm32(0U), atomic_body.Imm32(3U), {})};
        auto* phi = &*blocks[4]->PrependNewInst(blocks[4]->begin(), IR::Opcode::Phi);
        phi->SetFlags(IR::Type::U32);
        phi->AddPhiOperand(blocks[3], atomic);
        phi->AddPhiOperand(blocks[2], IR::Value{0U});
        IR::IREmitter merge{*blocks[4]};
        if (restore_first) {
            merge.SetExec(merge.Imm1(true));
        }
        IR::U32 value{phi};
        if (undefined_source) {
            value = IR::U32{&*blocks[4]->PrependNewInst(blocks[4]->end(), IR::Opcode::UndefU32)};
        }
        IR::Value result;
        if (operation == IR::Opcode::Ballot) {
            result = merge.Ballot(merge.IEqual(invocation, merge.Imm32(0U)));
        } else if (operation == IR::Opcode::ReadFirstLane) {
            result = merge.ReadFirstLane(value);
        } else {
            result = merge.ReadLane(value, merge.Imm32(0U));
        }
        selected = result.Inst();
        merge.Reference(result);
        merge.SetExec(merge.Imm1(true));

        const auto block_node = [&](IR::Block* block) {
            auto& node = program.syntax_list.emplace_back();
            node.type = IR::AbstractSyntaxNode::Type::Block;
            node.data.block = block;
        };
        const auto if_node = [&](const IR::U1& condition, IR::Block* body, IR::Block* target) {
            auto& node = program.syntax_list.emplace_back();
            node.type = IR::AbstractSyntaxNode::Type::If;
            node.data.if_node.cond = condition;
            node.data.if_node.body = body;
            node.data.if_node.merge = target;
        };
        const auto end_node = [&](IR::Block* target) {
            auto& node = program.syntax_list.emplace_back();
            node.type = IR::AbstractSyntaxNode::Type::EndIf;
            node.data.end_if.merge = target;
        };
        block_node(blocks[0]);
        if_node(first, blocks[1], blocks[2]);
        block_node(blocks[1]);
        end_node(blocks[2]);
        block_node(blocks[2]);
        if_node(second, blocks[3], blocks[4]);
        block_node(blocks[3]);
        end_node(blocks[4]);
        block_node(blocks[4]);
        program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    }

    void Lower() {
        Optimization::LowerLaneIdToLocalInvocationIndexPass(program, runtime, profile);
    }

    size_t Count(const IR::Block& block, IR::Opcode opcode) const {
        return std::ranges::count_if(
            block, [&](const IR::Inst& inst) { return inst.GetOpcode() == opcode; });
    }

    void ExpectNative(IR::Opcode opcode = IR::Opcode::ReadLane) const {
        EXPECT_EQ(Count(*blocks[4], opcode), 1U);
        EXPECT_EQ(Count(*blocks[4], IR::Opcode::Barrier), 0U);
        EXPECT_EQ(info.workgroup_scratch_memory_size, 0U);
    }
};

TEST(ChainedWaveReadLane, SecondClosedDiamondBroadcastsDespitePartialGuestExec) {
    ChainedWaveFixture fixture;
    fixture.Lower();
    EXPECT_EQ(fixture.Count(*fixture.blocks[4], IR::Opcode::ReadLane), 0U);
    EXPECT_EQ(fixture.Count(*fixture.blocks[4], IR::Opcode::Barrier), 2U);
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 256U);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(fixture.Count(*fixture.blocks[i], IR::Opcode::Barrier), 0U);
    }
}

TEST(ChainedWaveReadLane, AnExplicitFullRestoreAlsoAdmitsTheSecondMerge) {
    ChainedWaveFixture fixture{IR::Opcode::ReadLane, false, true};
    fixture.Lower();
    EXPECT_EQ(fixture.Count(*fixture.blocks[4], IR::Opcode::ReadLane), 0U);
    EXPECT_EQ(fixture.Count(*fixture.blocks[4], IR::Opcode::Barrier), 2U);
}

TEST(ChainedWaveReadLane, FirstMergeBallotAndSecondMergeAtomicBroadcastMaterializeTogether) {
    ChainedWaveFixture fixture;
    auto* header = fixture.blocks[2];
    auto point = header->begin();
    ++point; // After this merge's full EXEC restore, before its lane-zero selection.
    IR::IREmitter ir{*header, point};
    const auto invocation = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const auto ballot = ir.Ballot(ir.ILessThan(invocation, ir.Imm32(40U), false));
    const auto count = ir.IAdd(ir.BitCount(IR::U32{ir.CompositeExtract(ballot, 0)}),
                               ir.BitCount(IR::U32{ir.CompositeExtract(ballot, 1)}));
    for (IR::Inst& inst : *fixture.blocks[3]) {
        if (inst.GetOpcode() == IR::Opcode::BufferAtomicIAdd32) {
            inst.SetArg(2, count);
        }
    }
    fixture.Lower();
    EXPECT_EQ(fixture.Count(*header, IR::Opcode::Ballot), 0U);
    EXPECT_EQ(fixture.Count(*header, IR::Opcode::Barrier), 3U);
    EXPECT_EQ(fixture.Count(*fixture.blocks[4], IR::Opcode::ReadLane), 0U);
    EXPECT_EQ(fixture.Count(*fixture.blocks[4], IR::Opcode::Barrier), 2U);
    EXPECT_EQ(fixture.info.workgroup_scratch_memory_size, 256U);
}

TEST(ChainedWaveReadLane, PartialExecDoesNotGrantBallotOrReadFirstLaneAnException) {
    for (auto opcode : {IR::Opcode::Ballot, IR::Opcode::ReadFirstLane}) {
        ChainedWaveFixture fixture{opcode};
        fixture.Lower();
        fixture.ExpectNative(opcode);
    }
}

TEST(ChainedWaveReadLane, UndefinedSourcesAndDynamicSelectorsStayNative) {
    ChainedWaveFixture undefined{IR::Opcode::ReadLane, true};
    undefined.Lower();
    undefined.ExpectNative();
    ChainedWaveFixture dynamic;
    IR::IREmitter ir{*dynamic.blocks[0], dynamic.blocks[0]->begin()};
    dynamic.selected->SetArg(1, ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0));
    dynamic.Lower();
    dynamic.ExpectNative();
}

TEST(ChainedWaveReadLane, SideEntryOrExitInsideAnEarlierDiamondInvalidatesThePrefix) {
    for (bool side_entry : {false, true}) {
        ChainedWaveFixture fixture;
        if (side_entry) {
            fixture.blocks[0]->AddBranch(fixture.blocks[4]);
        } else {
            fixture.blocks[1]->AddBranch(fixture.blocks[4]);
        }
        fixture.Lower();
        fixture.ExpectNative();
    }
}

TEST(ChainedWaveReadLane, LoopsAndUnbalancedEarlierSyntaxStayNative) {
    for (bool loop : {false, true}) {
        ChainedWaveFixture fixture;
        fixture.program.syntax_list[3].type =
            loop ? IR::AbstractSyntaxNode::Type::Loop : IR::AbstractSyntaxNode::Type::Return;
        fixture.Lower();
        fixture.ExpectNative();
    }
}

struct ChainedGpuRecord {
    u32 reserved_counter;
    u32 old_base;
    u32 mask_low;
    u32 mask_high;
};
using ChainedGpuOutput = std::array<ChainedGpuRecord, 64>;

std::vector<u32> EmitChainedAtomicProbe(bool lower_wave) {
    ChainedWaveFixture fixture;
    fixture.info.flattened_ud_buf.resize(4);
    const auto descriptor = AmdGpu::Buffer::Null();
    std::memcpy(fixture.info.flattened_ud_buf.data(), &descriptor, sizeof(descriptor));
    fixture.info.buffers.push_back({
        .sharp_idx = 0,
        .used_types = IR::Type::U32,
        .buffer_type = BufferType::Guest,
        .is_written = true,
    });
    IR::IREmitter entry{*fixture.blocks[0], fixture.blocks[0]->begin()};
    entry.Prologue();
    const auto invocation = entry.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const auto lane_zero = entry.IEqual(invocation, entry.Imm32(0U));
    for (IR::Inst& inst : *fixture.blocks[0]) {
        if (inst.GetOpcode() == IR::Opcode::ConditionRef) {
            inst.SetArg(0, lane_zero);
        }
    }
    // Exactly one invocation initializes the counter. Both memory and execution barriers
    // occur at the first closed merge, never inside either lane-zero branch.
    IR::IREmitter initializer{*fixture.blocks[1]};
    initializer.StoreBufferU32(1, initializer.Imm32(0U), initializer.Imm32(0U),
                               initializer.Imm32(17U), {});
    auto point = fixture.blocks[2]->begin();
    ++point;
    IR::IREmitter merge{*fixture.blocks[2], point};
    merge.WorkgroupMemoryBarrier();
    merge.Barrier();
    const auto ballot = merge.Ballot(merge.ILessThan(invocation, merge.Imm32(40U), false));
    const IR::U32 low{merge.CompositeExtract(ballot, 0)};
    const IR::U32 high{merge.CompositeExtract(ballot, 1)};
    const auto count = merge.IAdd(merge.BitCount(low), merge.BitCount(high));
    const auto reservation = merge.IMul(count, merge.Imm32(3U));
    for (IR::Inst& inst : *fixture.blocks[3]) {
        if (inst.GetOpcode() == IR::Opcode::BufferAtomicIAdd32) {
            inst.SetArg(2, reservation);
        }
    }
    IR::IREmitter output{*fixture.blocks[4]};
    const auto record = output.IMul(invocation, output.Imm32(4U));
    const std::array values{IR::U32{fixture.selected}, low, high};
    for (u32 component = 0; component < values.size(); ++component) {
        output.StoreBufferU32(1, output.Imm32(0U),
                              output.IAdd(record, output.Imm32(component + 1U)), values[component],
                              {});
    }
    output.Epilogue();
    fixture.program.post_order_blocks = IR::PostOrder(fixture.program.syntax_list.front());
    Optimization::SsaRewritePass(fixture.program);
    Optimization::IdentityRemovalPass(fixture.program.blocks);
    Optimization::ConstantPropagationPass(fixture.program.post_order_blocks);
    Optimization::DeadCodeEliminationPass(fixture.program, true);
    if (lower_wave) {
        fixture.Lower();
        EXPECT_EQ(fixture.Count(*fixture.blocks[2], IR::Opcode::Ballot), 0U);
        EXPECT_EQ(fixture.Count(*fixture.blocks[4], IR::Opcode::ReadLane), 0U);
        EXPECT_EQ(fixture.Count(*fixture.blocks[2], IR::Opcode::Barrier), 4U);
        EXPECT_EQ(fixture.Count(*fixture.blocks[4], IR::Opcode::Barrier), 2U);
        EXPECT_EQ(fixture.Count(*fixture.blocks[1], IR::Opcode::Barrier), 0U);
        EXPECT_EQ(fixture.Count(*fixture.blocks[3], IR::Opcode::Barrier), 0U);
    }
    Optimization::IdentityRemovalPass(fixture.program.blocks);
    Optimization::DeadCodeEliminationPass(fixture.program);
    Optimization::CollectShaderInfoPass(fixture.program, fixture.profile);
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.runtime, fixture.program, bindings);
}

class ChainedWaveGpu : public testing::Test {
protected:
    static void TearDownTestSuite() {
        gcn_test::Runner::DestroyInstance();
    }
};

TEST_F(ChainedWaveGpu, NativeSubgroupsCannotSupplyTheSameFullMaskAndAtomicBase) {
    auto result = gcn_test::Runner::subgroup32_instance();
    if (!result && result.error().code == gcn_test::Error::NoSuitableDevice) {
        GTEST_SKIP() << result.error().message;
    }
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_EQ((*result)->effective_compute_subgroup_size(), 32U);
    const auto output = (*result)->run<ChainedGpuOutput>(EmitChainedAtomicProbe(false));
    ASSERT_TRUE(output.has_value()) << output.error().message;
    EXPECT_TRUE(std::ranges::any_of(*output, [](const ChainedGpuRecord& record) {
        return record.old_base != 17U || record.mask_low != 0xFFFFFFFFU ||
               record.mask_high != 0xFFU;
    }));
}

TEST_F(ChainedWaveGpu, TwoReconvergedRegionsShareOneAtomicReservationForOneHundredRuns) {
    auto result = gcn_test::Runner::subgroup32_instance();
    if (!result && result.error().code == gcn_test::Error::NoSuitableDevice) {
        GTEST_SKIP() << result.error().message;
    }
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_EQ((*result)->effective_compute_subgroup_size(), 32U);
    const auto spirv = EmitChainedAtomicProbe(true);
    for (u32 repeat = 0; repeat < 100U; ++repeat) {
        const auto output = (*result)->run<ChainedGpuOutput>(spirv);
        ASSERT_TRUE(output.has_value()) << output.error().message;
        EXPECT_EQ((*output)[0].reserved_counter, 137U);
        for (u32 lane = 0; lane < 64U; ++lane) {
            SCOPED_TRACE(lane);
            EXPECT_EQ((*output)[lane].old_base, 17U);
            EXPECT_EQ((*output)[lane].mask_low, 0xFFFFFFFFU);
            EXPECT_EQ((*output)[lane].mask_high, 0xFFU);
        }
    }
}

} // namespace
