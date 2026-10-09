// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <optional>
#include <gtest/gtest.h>

#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"

namespace {
using namespace Shader;

struct MbcntFixture {
    explicit MbcntFixture(std::array<u32, 3> shape = {64, 1, 1}, Stage physical = Stage::Compute,
                          LogicalStage logical = LogicalStage::Compute,
                          Stage runtime_stage = Stage::Compute)
        : program{info} {
        info.stage = physical;
        info.l_stage = logical;
        runtime.Initialize(runtime_stage);
        runtime.cs_info.workgroup_size = shape;
        profile.subgroup_size = 64;
        block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        auto& syntax = program.syntax_list.emplace_back();
        syntax.type = IR::AbstractSyntaxNode::Type::Block;
        syntax.data.block = block;
        program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
        program.post_order_blocks = IR::PostOrder(program.syntax_list.front());
    }

    IR::Value Emit(bool low, u32 mask, u32 addend, bool inline_all_ones = false,
                   bool high_then_low = false, u32 high_mask = 0U,
                   bool preserve_native_control = false) {
        Gcn::Translator translator{info, runtime, profile};
        translator.EmitPrologue(block);
        IR::IREmitter ir{*block};
        ir.SetScalarReg(IR::ScalarReg::S4, ir.Imm32(mask));
        ir.SetScalarReg(IR::ScalarReg::S5, ir.Imm32(high_mask));
        ir.SetVectorReg(IR::VectorReg::V1, ir.Imm32(addend));
        Gcn::GcnInst inst{};
        inst.src[0].field =
            inline_all_ones ? Gcn::OperandField::SignedConstIntNeg : Gcn::OperandField::ScalarGPR;
        inst.src[0].code = inline_all_ones ? 193U : 4U;
        inst.src[1].field = Gcn::OperandField::VectorGPR;
        inst.src[1].code = 1;
        inst.dst[0].field = Gcn::OperandField::VectorGPR;
        inst.dst[0].code = 1;
        if (high_then_low) {
            inst.src[0].code = 5;
            translator.V_MBCNT_U32_B32(false, inst);
            inst.src[0].code = 4;
        }
        translator.V_MBCNT_U32_B32(low, inst);
        const auto value = ir.GetVectorReg(IR::VectorReg::V1);
        ir.Reference(value);
        if (preserve_native_control) {
            ir.Reference(ir.LaneId());
        }
        Optimization::SsaRewritePass(program);
        for (auto& instruction : *block) {
            if (instruction.GetOpcode() == IR::Opcode::Reference) {
                return instruction.Arg(0);
            }
        }
        ADD_FAILURE() << "Missing translated MBCNT output";
        return {};
    }

    Info info{};
    IR::Program program;
    Pools pools{};
    RuntimeInfo runtime{};
    Profile profile{};
    IR::Block* block{};
};

std::optional<u32> EvaluateMbcnt(const IR::Value& value, u32 lane, u32 depth = 0,
                                 std::array<u32, 3> shape = {64, 1, 1},
                                 bool native_subgroup32 = false) {
    if (depth > 64) {
        return std::nullopt;
    }
    if (value.IsImmediate()) {
        return value.Type() == IR::Type::U1 ? u32(value.U1()) : value.U32();
    }
    const auto* const inst = value.TryInstRecursive();
    if (!inst) {
        return std::nullopt;
    }
    if (inst->GetOpcode() == IR::Opcode::LaneId) {
        return native_subgroup32 ? lane & 31U : lane;
    }
    if (inst->GetOpcode() == IR::Opcode::GetAttributeU32 &&
        inst->Arg(0).Attribute() == IR::Attribute::LocalInvocationId) {
        const std::array local{lane % shape[0], (lane / shape[0]) % shape[1],
                               lane / (shape[0] * shape[1])};
        return local[inst->Arg(1).U32()];
    }
    std::array<u32, 3> args{};
    if (inst->NumArgs() > args.size()) {
        return std::nullopt;
    }
    for (u32 i = 0; i < inst->NumArgs(); ++i) {
        const auto arg = EvaluateMbcnt(inst->Arg(i), lane, depth + 1, shape, native_subgroup32);
        if (!arg) {
            return std::nullopt;
        }
        args[i] = *arg;
    }
    switch (inst->GetOpcode()) {
    case IR::Opcode::IAdd32:
        return args[0] + args[1];
    case IR::Opcode::ISub32:
        return args[0] - args[1];
    case IR::Opcode::IMul32:
        return args[0] * args[1];
    case IR::Opcode::BitwiseAnd32:
        return args[0] & args[1];
    case IR::Opcode::BitwiseOr32:
        return args[0] | args[1];
    case IR::Opcode::BitwiseNot32:
        return ~args[0];
    case IR::Opcode::ShiftLeftLogical32:
        return args[0] << (args[1] & 31U);
    case IR::Opcode::ShiftRightLogical32:
        return args[0] >> (args[1] & 31U);
    case IR::Opcode::BitCount32:
        return std::popcount(args[0]);
    case IR::Opcode::ULessThan32:
        return u32(args[0] < args[1]);
    case IR::Opcode::ULessThanEqual32:
        return u32(args[0] <= args[1]);
    case IR::Opcode::UGreaterThan32:
        return u32(args[0] > args[1]);
    case IR::Opcode::UGreaterThanEqual32:
        return u32(args[0] >= args[1]);
    case IR::Opcode::IEqual32:
        return u32(args[0] == args[1]);
    case IR::Opcode::SelectU32:
        return args[0] ? args[1] : args[2];
    default:
        return std::nullopt;
    }
}

u32 BelowHalf(u32 lane, bool low) {
    const u32 count = low ? std::min(lane, 32U) : (lane > 32U ? lane - 32U : 0U);
    return count == 32U ? ~0U : ((1U << count) - 1U);
}

TEST(MbcntInstruction, LowHalfCountsSparseMaskAndPreservesAccumulator) {
    for (const u32 mask : {0x80000005U, 0x55555555U, 0xF0000000U}) {
        MbcntFixture fixture;
        const auto output = fixture.Emit(true, mask, 7U);
        for (u32 lane = 0; lane < 64; ++lane) {
            SCOPED_TRACE(lane);
            EXPECT_EQ(EvaluateMbcnt(output, lane),
                      7U + u32(std::popcount(mask & BelowHalf(lane, true))));
        }
    }
}

TEST(MbcntInstruction, HighHalfCountsOnlyBitsBelowTheUpperGuestLane) {
    MbcntFixture fixture;
    constexpr u32 mask = 0x80000005U;
    const auto output = fixture.Emit(false, mask, 9U);
    for (u32 lane = 0; lane < 64; ++lane) {
        SCOPED_TRACE(lane);
        EXPECT_EQ(EvaluateMbcnt(output, lane),
                  9U + u32(std::popcount(mask & BelowHalf(lane, false))));
    }
}

TEST(MbcntInstruction, HighThenLowBuildsTheCompleteSparseWavePrefix) {
    MbcntFixture fixture;
    constexpr u32 low_mask = 0x80000005U;
    constexpr u32 high_mask = 0x8000000BU;
    const auto output = fixture.Emit(true, low_mask, 3U, false, true, high_mask);
    for (u32 lane = 0; lane < 64; ++lane) {
        SCOPED_TRACE(lane);
        EXPECT_EQ(EvaluateMbcnt(output, lane),
                  3U + u32(std::popcount(low_mask & BelowHalf(lane, true))) +
                      u32(std::popcount(high_mask & BelowHalf(lane, false))));
    }
}

TEST(MbcntInstruction, InlineAllOnesLowHalfDoesNotDiscardItsAccumulator) {
    MbcntFixture fixture;
    const auto output = fixture.Emit(true, ~0U, 11U, true);
    for (u32 lane = 0; lane < 64; ++lane) {
        SCOPED_TRACE(lane);
        EXPECT_EQ(EvaluateMbcnt(output, lane), 11U + std::min(lane, 32U));
    }
}

TEST(MbcntInstruction, EmptyMasksPreserveTheAccumulator) {
    for (const bool low : {false, true}) {
        MbcntFixture fixture;
        const auto output = fixture.Emit(low, 0U, 0xFFFFFFFEU);
        for (const u32 lane : {0U, 1U, 31U, 32U, 63U}) {
            EXPECT_EQ(EvaluateMbcnt(output, lane), 0xFFFFFFFEU);
        }
    }
}

TEST(MbcntInstruction, ExactWaveComputeArithmeticDoesNotUseAnUnrelatedNativeLaneIdentity) {
    constexpr u32 low_mask = 0x80000005U;
    constexpr u32 high_mask = 0x8000000BU;
    for (const auto shape :
         {std::array{64U, 1U, 1U}, std::array{8U, 8U, 1U}, std::array{4U, 4U, 4U}}) {
        MbcntFixture fixture{shape};
        fixture.profile.subgroup_size = 32;
        const auto output = fixture.Emit(true, low_mask, 3U, false, true, high_mask, true);
        u32 native_lane_instructions{};
        for (const auto& inst : *fixture.block) {
            native_lane_instructions += inst.GetOpcode() == IR::Opcode::LaneId;
        }
        // Only the independent native control remains; no global LaneId rewrite is requested.
        EXPECT_EQ(native_lane_instructions, 1U);
        for (u32 lane = 0; lane < 64; ++lane) {
            SCOPED_TRACE(lane);
            EXPECT_EQ(EvaluateMbcnt(output, lane, 0, shape, true),
                      3U + u32(std::popcount(low_mask & BelowHalf(lane, true))) +
                          u32(std::popcount(high_mask & BelowHalf(lane, false))));
        }
    }
}

TEST(MbcntInstruction, OtherStageOrUnprovenWorkgroupKeepsItsExistingLaneSource) {
    for (const auto shape : {std::array{128U, 1U, 1U}, std::array{0U, 1U, 1U},
                             std::array{0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU}}) {
        MbcntFixture fixture{shape};
        const auto output = fixture.Emit(false, 7U, 2U);
        EXPECT_EQ(EvaluateMbcnt(output, 40U, 0, {64, 1, 1}, true), 2U);
    }
    for (const u32 variant : {0U, 1U, 2U}) {
        MbcntFixture fixture{{64, 1, 1},
                             variant == 0 ? Stage::Vertex : Stage::Compute,
                             variant == 1 ? LogicalStage::Vertex : LogicalStage::Compute,
                             variant == 2 ? Stage::Vertex : Stage::Compute};
        const auto output = fixture.Emit(false, 7U, 2U);
        EXPECT_EQ(EvaluateMbcnt(output, 40U, 0, {64, 1, 1}, true), 2U);
    }
}
} // namespace
