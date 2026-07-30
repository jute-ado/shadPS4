// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cmath>
#include <span>

#include <gtest/gtest.h>
#include <half.hpp>
#include <spirv/unified1/spirv.hpp11>

#include "gcn_test_runner.hpp"
#include "instructions.hpp"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/recompiler.h"
#include "translator.hpp"

class GcnTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    static void TearDownTestSuite() {
        gcn_test::Runner::DestroyInstance();
    }
};

struct F32x2 {
    float a;
    float b;
};

namespace {

size_t CountSpirvOpcode(std::span<const u32> spirv, spv::Op opcode) {
    constexpr size_t SpirvHeaderWords = 5;
    size_t count{};
    for (size_t offset = SpirvHeaderWords; offset < spirv.size();) {
        const u32 instruction = spirv[offset];
        const size_t word_count = instruction >> 16;
        if (word_count == 0 || word_count > spirv.size() - offset) {
            return 0;
        }
        if ((instruction & 0xffffU) == static_cast<u32>(opcode)) {
            ++count;
        }
        offset += word_count;
    }
    return count;
}

bool ContainsSpirvOpcode(std::span<const u32> spirv, spv::Op opcode) {
    return CountSpirvOpcode(spirv, opcode) != 0;
}

constexpr u64 MakeDsSwizzle(u8 source, u8 destination, u8 offset0, u8 offset1) {
    return u64{offset0} | (u64{offset1} << 8) |
           (u64{std::to_underlying(Shader::Gcn::OpcodeDS::DS_SWIZZLE_B32)} << 18) |
           u64{std::to_underlying(Shader::Gcn::InstEncoding::DS)} | (u64{source} << 32) |
           (u64{destination} << 56);
}

} // Anonymous namespace

TEST(GcnIrPass, devirtualizes_wave_serialized_vgpr_selector) {
    Shader::Info info{};
    Shader::Pools pools;
    Shader::IR::Program program{info};
    Shader::IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U32 per_lane_index =
        ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U32 first_index = ir.ReadFirstLane(per_lane_index);
    Shader::IR::Inst* const read_first = first_index.Inst();

    const Shader::IR::U1 selected_lane = ir.IEqual(first_index, per_lane_index);
    Shader::IR::Inst* const selected_lane_compare = selected_lane.Inst();
    const Shader::IR::U1 active_selected_lane = ir.LogicalAnd(ir.Imm1(true), selected_lane);
    static_cast<void>(ir.ConditionRef(active_selected_lane));
    const Shader::IR::U1 remaining_lanes =
        ir.LogicalAnd(ir.Imm1(true), ir.LogicalNot(active_selected_lane));
    static_cast<void>(ir.ConditionRef(remaining_lanes));

    const Shader::IR::U1 index_zero = ir.IEqual(first_index, ir.Imm32(0));
    const Shader::IR::U1 index_one = ir.IEqual(first_index, ir.Imm32(1));
    const Shader::IR::U1 index_two = ir.IEqual(first_index, ir.Imm32(2));
    Shader::IR::Inst* const index_zero_compare = index_zero.Inst();
    static_cast<void>(
        ir.Select(index_zero, ir.Imm32(10), ir.Select(index_one, ir.Imm32(11), ir.Imm32(12))));
    static_cast<void>(ir.Select(index_two, ir.Imm32(20), ir.Imm32(21)));

    Shader::Optimization::WaveSerializedVgprIndexPass(program);

    EXPECT_FALSE(read_first->HasUses());
    EXPECT_FALSE(selected_lane_compare->HasUses());
    EXPECT_EQ(index_zero_compare->Arg(0).Resolve(), per_lane_index.Resolve());
}

TEST(GcnIrPass, devirtualizes_wave_serialized_selector_with_equivalent_phi_masks) {
    Shader::Info info{};
    Shader::Pools pools;
    Shader::IR::Program program{info};
    Shader::IR::Block* const entry = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::Block* const backedge = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {entry, backedge, block};

    auto active_mask_it =
        block->PrependNewInst(block->end(), Shader::IR::Opcode::Phi);
    Shader::IR::Inst* const active_mask_phi = &*active_mask_it;
    active_mask_phi->SetFlags(Shader::IR::Type::U1);
    active_mask_phi->AddPhiOperand(entry, Shader::IR::Value{true});
    active_mask_phi->AddPhiOperand(backedge, Shader::IR::Value{false});

    auto remaining_mask_it =
        block->PrependNewInst(block->end(), Shader::IR::Opcode::Phi);
    Shader::IR::Inst* const remaining_mask_phi = &*remaining_mask_it;
    remaining_mask_phi->SetFlags(Shader::IR::Type::U1);
    remaining_mask_phi->AddPhiOperand(entry, Shader::IR::Value{true});
    remaining_mask_phi->AddPhiOperand(backedge, Shader::IR::Value{false});

    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U32 per_lane_index =
        ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U32 first_index = ir.ReadFirstLane(per_lane_index);
    Shader::IR::Inst* const read_first = first_index.Inst();

    const Shader::IR::U1 selected_lane = ir.IEqual(first_index, per_lane_index);
    const Shader::IR::U1 active_selected_lane =
        ir.LogicalAnd(Shader::IR::U1{active_mask_phi}, selected_lane);
    static_cast<void>(ir.ConditionRef(active_selected_lane));
    const Shader::IR::U1 remaining_lanes =
        ir.LogicalAnd(Shader::IR::U1{remaining_mask_phi}, ir.LogicalNot(active_selected_lane));
    static_cast<void>(ir.ConditionRef(remaining_lanes));
    active_mask_phi->SetArg(1, remaining_lanes);
    remaining_mask_phi->SetArg(1, remaining_lanes);

    const Shader::IR::U1 index_zero = ir.IEqual(first_index, ir.Imm32(0));
    const Shader::IR::U1 index_one = ir.IEqual(first_index, ir.Imm32(1));
    static_cast<void>(ir.Select(index_zero, ir.Imm32(10), ir.Imm32(11)));
    static_cast<void>(ir.Select(index_one, ir.Imm32(20), ir.Imm32(21)));

    Shader::Optimization::WaveSerializedVgprIndexPass(program);

    EXPECT_FALSE(read_first->HasUses());
}

TEST(GcnIrPass, preserves_selector_without_remaining_lane_loop) {
    Shader::Info info{};
    Shader::Pools pools;
    Shader::IR::Program program{info};
    Shader::IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U32 per_lane_index =
        ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U32 first_index = ir.ReadFirstLane(per_lane_index);
    Shader::IR::Inst* const read_first = first_index.Inst();

    const Shader::IR::U1 selected_lane = ir.IEqual(first_index, per_lane_index);
    const Shader::IR::U1 active_selected_lane = ir.LogicalAnd(ir.Imm1(true), selected_lane);
    static_cast<void>(ir.ConditionRef(active_selected_lane));
    const Shader::IR::U1 index_zero = ir.IEqual(first_index, ir.Imm32(0));
    const Shader::IR::U1 index_one = ir.IEqual(first_index, ir.Imm32(1));
    static_cast<void>(ir.Select(index_zero, ir.Imm32(10), ir.Imm32(11)));
    static_cast<void>(ir.Select(index_one, ir.Imm32(20), ir.Imm32(21)));

    Shader::Optimization::WaveSerializedVgprIndexPass(program);

    EXPECT_TRUE(read_first->HasUses());
}

TEST(GcnIrPass, preserves_selector_with_mismatched_remaining_lane_mask) {
    Shader::Info info{};
    Shader::Pools pools;
    Shader::IR::Program program{info};
    Shader::IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U32 per_lane_index =
        ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U32 first_index = ir.ReadFirstLane(per_lane_index);
    Shader::IR::Inst* const read_first = first_index.Inst();

    const Shader::IR::U1 active_mask = ir.IEqual(per_lane_index, ir.Imm32(0));
    const Shader::IR::U1 selected_lane = ir.IEqual(first_index, per_lane_index);
    const Shader::IR::U1 active_selected_lane = ir.LogicalAnd(active_mask, selected_lane);
    static_cast<void>(ir.ConditionRef(active_selected_lane));
    const Shader::IR::U1 different_mask = ir.IEqual(per_lane_index, ir.Imm32(1));
    const Shader::IR::U1 remaining_lanes =
        ir.LogicalAnd(different_mask, ir.LogicalNot(active_selected_lane));
    static_cast<void>(ir.ConditionRef(remaining_lanes));

    const Shader::IR::U1 index_zero = ir.IEqual(first_index, ir.Imm32(0));
    const Shader::IR::U1 index_one = ir.IEqual(first_index, ir.Imm32(1));
    static_cast<void>(ir.Select(index_zero, ir.Imm32(10), ir.Imm32(11)));
    static_cast<void>(ir.Select(index_one, ir.Imm32(20), ir.Imm32(21)));

    Shader::Optimization::WaveSerializedVgprIndexPass(program);

    EXPECT_TRUE(read_first->HasUses());
}

TEST(GcnIrPass, preserves_selector_with_semantic_loop_mask_use) {
    Shader::Info info{};
    Shader::Pools pools;
    Shader::IR::Program program{info};
    Shader::IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U32 per_lane_index =
        ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U32 first_index = ir.ReadFirstLane(per_lane_index);
    Shader::IR::Inst* const read_first = first_index.Inst();

    const Shader::IR::U1 selected_lane = ir.IEqual(first_index, per_lane_index);
    const Shader::IR::U1 active_selected_lane = ir.LogicalAnd(ir.Imm1(true), selected_lane);
    static_cast<void>(ir.ConditionRef(active_selected_lane));
    const Shader::IR::U1 remaining_lanes =
        ir.LogicalAnd(ir.Imm1(true), ir.LogicalNot(active_selected_lane));
    static_cast<void>(ir.ConditionRef(remaining_lanes));
    static_cast<void>(ir.Select(active_selected_lane, ir.Imm32(30), ir.Imm32(31)));

    const Shader::IR::U1 index_zero = ir.IEqual(first_index, ir.Imm32(0));
    const Shader::IR::U1 index_one = ir.IEqual(first_index, ir.Imm32(1));
    static_cast<void>(ir.Select(index_zero, ir.Imm32(10), ir.Imm32(11)));
    static_cast<void>(ir.Select(index_one, ir.Imm32(20), ir.Imm32(21)));

    Shader::Optimization::WaveSerializedVgprIndexPass(program);

    EXPECT_TRUE(read_first->HasUses());
}

TEST(GcnIrPass, preserves_semantic_read_first_lane) {
    Shader::Info info{};
    Shader::Pools pools;
    Shader::IR::Program program{info};
    Shader::IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    const Shader::IR::U32 per_lane_value =
        ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U32 first_value = ir.ReadFirstLane(per_lane_value);
    Shader::IR::Inst* const read_first = first_value.Inst();
    static_cast<void>(ir.IAdd(first_value, ir.Imm32(1)));

    Shader::Optimization::WaveSerializedVgprIndexPass(program);

    EXPECT_TRUE(read_first->HasUses());
}

TEST_F(GcnTest, dma_fault_bits_are_marked_atomically) {
    // s_load_dword s4, s[0:1], 0
    constexpr u64 load_dword = 0xc0020100U;
    const std::array<u64, 2> instructions{
        load_dword,
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S4).Get(),
    };

    const auto spirv = TranslateToSpirvWithDma(instructions);

    EXPECT_TRUE(ContainsSpirvOpcode(spirv, spv::Op::OpAtomicOr));
}

TEST_F(GcnTest, fixed_wave64_readlane_uses_workgroup_exchange_on_wave32_hosts) {
    for (const u32 lane : {0U, 32U}) {
        const auto lane_operand = static_cast<VOperand8>(std::to_underlying(SOperand9::Const0) + lane);
        const std::array<u64, 2> instructions{
            VOP2(OpcodeVOP2::V_READLANE_B32, VOperand8::V0, SOperand9::V0, lane_operand).Get(),
            VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S0).Get(),
        };

        const auto spirv = TranslateToSpirv(instructions, {64, 1, 1});

        EXPECT_FALSE(ContainsSpirvOpcode(spirv, spv::Op::OpGroupNonUniformBroadcast))
            << "guest lane " << lane;
        EXPECT_EQ(CountSpirvOpcode(spirv, spv::Op::OpControlBarrier), 2)
            << "guest lane " << lane;

        if (lane == 32) {
            auto runner = gcn_test::Runner::instance().value();
            const auto result = runner->run<u32>(spirv);
            ASSERT_TRUE(result.has_value());
            EXPECT_EQ(*result, lane);
        }
    }
}

TEST_F(GcnTest, adjacent_fixed_wave64_readlanes_share_one_barrier_pair) {
    const std::array<u64, 4> instructions{
        VOP2(OpcodeVOP2::V_READLANE_B32, VOperand8::V0, SOperand9::V0,
             static_cast<VOperand8>(std::to_underlying(SOperand9::Const0)))
            .Get(),
        VOP2(OpcodeVOP2::V_READLANE_B32, VOperand8::V1, SOperand9::V0,
             static_cast<VOperand8>(std::to_underlying(SOperand9::Const32)))
            .Get(),
        SOP2(OpcodeSOP2::S_ADD_U32, SOperand7::S0, SOperand8::S0, SOperand8::S1).Get(),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S0).Get(),
    };

    const auto spirv = TranslateToSpirv(instructions, {64, 1, 1});

    EXPECT_EQ(CountSpirvOpcode(spirv, spv::Op::OpGroupNonUniformBroadcast), 0);
    EXPECT_EQ(CountSpirvOpcode(spirv, spv::Op::OpControlBarrier), 2);

    auto runner = gcn_test::Runner::instance().value();
    const auto result = runner->run<u32>(spirv);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 32U);
}

TEST_F(GcnTest, ds_swizzle_uses_nonuniform_subgroup_shuffle) {
    // DS swizzle computes a different source lane for each invocation. SPIR-V broadcast requires
    // its invocation id to be dynamically uniform, while shuffle permits this lane permutation.
    constexpr u8 and_mask = 0x1f;
    constexpr u8 xor_one = 0x04;
    const std::array<u64, 2> instructions{
        MakeDsSwizzle(0, 1, and_mask, xor_one),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::V1).Get(),
    };

    const auto spirv = TranslateToSpirv(instructions);

    EXPECT_TRUE(ContainsSpirvOpcode(spirv, spv::Op::OpGroupNonUniformShuffleXor));
    EXPECT_FALSE(ContainsSpirvOpcode(spirv, spv::Op::OpGroupNonUniformBroadcast));
}

// Example
// TEST_F(GcnTest, test_name) {
//     // Runner sets the vulkan context
//     auto runner = gcn_test::Runner::instance().value();
//
//     // v_add_f32 v0, v0, v1
//     auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_ADD_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
//
//     // run<T> tells how to interpret the result (only 32bit as of now)
//     // the second argument is templated, it can be at most 4 u32s
//     // the data is accessible by the instruction in v0-4 and s0-4 (mirrored)
//     // the result has to be placed in v0
//     auto result = runner->run<float>(spirv, F32x2{1.5f, 6.0f});
//
//     EXPECT_TRUE(result.has_value());
//     EXPECT_EQ(*result, 7.5f);
// }

TEST_F(GcnTest, add_f32) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_ADD_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<float>(spirv, F32x2{1.5f, 6.0f});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 7.5f);
}

TEST_F(GcnTest, add_i32_carry_feeds_addc_u32) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 2> instructions{
        VOP2(OpcodeVOP2::V_ADD_I32, VOperand8::V1, SOperand9::S0, VOperand8::V1).Get(),
        VOP2(OpcodeVOP2::V_ADDC_U32, VOperand8::V0, SOperand9::S2, VOperand8::V3).Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);

    const auto overflow = runner->run<u32>(spirv, std::array{0xffffffffU, 1U, 7U, 0U});
    ASSERT_TRUE(overflow.has_value());
    EXPECT_EQ(*overflow, 8U);

    const auto no_overflow = runner->run<u32>(spirv, std::array{2U, 1U, 7U, 0U});
    ASSERT_TRUE(no_overflow.has_value());
    EXPECT_EQ(*no_overflow, 7U);
}

TEST_F(GcnTest, add_nan) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_ADD_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<float>(spirv, F32x2{1.0f, std::numeric_limits<float>::quiet_NaN()});

    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(std::isnan(*result));
}

using half = half_float::half;

struct F16x2 {
    half a;
    half b = half(0.0f);

    bool operator==(const F16x2& rhs) const = default;
};

static_assert(sizeof(F16x2) == sizeof(float));

TEST_F(GcnTest, add_f16) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_ADD_F16, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f)}, F16x2{half(1.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, F16x2{half(2.0f)});
}

TEST_F(GcnTest, add_f16_clamp) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetClamp(true).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f)}, F16x2{half(1.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, F16x2{half(1.0f)}); //confirmed with neo
}

TEST_F(GcnTest, add_f16_neg) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetNeg({true, true, false}).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f)}, F16x2{half(1.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ((*result).a, half(-2.0f)); //confirmed with neo
}

TEST_F(GcnTest, add_f16_opsel_hi) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetOpSel({true, true, false, true}).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f), half(2.0f)}, F16x2{half(1.0f), half(2.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ((*result).a, half(1.0f));
    EXPECT_EQ((*result).b, half(4.0f));
}

TEST_F(GcnTest, sub_f16) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_SUB_F16, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(0.0f)}, F16x2{half(1.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, F16x2{half(-1.0f)}); //confirmed with neo
}

TEST_F(GcnTest, mul_legacy_nan) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_MUL_LEGACY_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<u32>(spirv, std::array{u32(0), u32(0x7fc00000)});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0);
}

TEST_F(GcnTest, mul_nan) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_MUL_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<float>(spirv, std::array{u32(0), u32(0x7fc00000)});

    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(std::isnan(*result));
}

TEST_F(GcnTest, min_legacy_nan) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_MIN_LEGACY_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<u32>(spirv, std::array{u32(0), u32(0x7fc00000)});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x7fc00000);
}

TEST_F(GcnTest, min_nan) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_MIN_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<float>(spirv, std::array{u32(0), u32(0x7fc00000)});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0);
}

TEST_F(GcnTest, add3_u32_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD3_U32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).Get());
    auto result = runner->run<u32>(spirv, std::array{0, 1, 2});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 3);
}

TEST_F(GcnTest, add3_u32_2) {
    auto runner = gcn_test::Runner::instance().value();
    auto big = 2000000000;

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD3_U32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).Get());
    auto result = runner->run<u32>(spirv, std::array{big, big, big});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x65A0BC00);
}

TEST_F(GcnTest, add3_u32_3) {
    auto runner = gcn_test::Runner::instance().value();
    auto big = 2000000000;

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD3_U32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetClamp(true).Get());
    auto result = runner->run<u32>(spirv, std::array{big, big, big});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x65A0BC00);
}

TEST_F(GcnTest, add3_u32_4) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD3_U32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetNeg({1,0,0}).Get());
    auto result = runner->run<u32>(spirv, std::array{0, 1, 2});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x80000003);
}

TEST_F(GcnTest, or3_u32_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_OR3_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0xF0F0F0F0, 0x07070707, 0x11111111});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xF7F7F7F7);
}

TEST_F(GcnTest, or3_u32_2) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_OR3_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).Get());
    auto result = runner->run<u32>(spirv, std::array{0x07070707, 0x11111111, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x57575757);
}

TEST_F(GcnTest, or3_u32_3) {
    auto runner = gcn_test::Runner::instance().value();
    auto big = 2000000000;

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_OR3_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetClamp(true).Get());
    auto result = runner->run<u32>(spirv, std::array{0x07070707, 0x11111111, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x57575757);
}

TEST_F(GcnTest, or3_u32_4) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_OR3_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetNeg({0,0,1}).Get());
    auto result = runner->run<u32>(spirv, std::array{0x07070707, 0x11111111, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xD7575757);
}

TEST_F(GcnTest, and_or_b32_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0xF0F0F0F0, 0x07070707, 0x11111111});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x11111111);
}

TEST_F(GcnTest, and_or_b32_2) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOmod(Omod::Mul2).Get());
    auto result = runner->run<u32>(spirv, std::array{0x40404040, 0x40404040, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x40404040);
}

TEST_F(GcnTest, and_or_b32_3) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetClamp(true).Get());
    auto result = runner->run<u32>(spirv, std::array{0x40404040, 0x40404040, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x40404040);
}

TEST_F(GcnTest, and_or_b32_4) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetNeg({1,0,0}).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0x07070707, 0x11111111, 0xF0F0F0F0});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xF1F1F1F1);
}

TEST_F(GcnTest, and_or_b32_5) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetNeg({1,0,0}).SetAbs({1,0,0}).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0x77777777, 0xB0B0B0B0, 0x11111111});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xB1313131);
}

TEST_F(GcnTest, and_or_b32_6) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOmod(Omod::Mul2).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0x40404040, 0xB0B0B0B0, 0x11111111});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x11111111);
}

TEST_F(GcnTest, and_or_b32_7) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOmod(Omod::Div2).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0xB0B0B0B0, 0x77777777, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x70707070);
}

TEST_F(GcnTest, and_or_b32_8) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetAbs({1,1,0}).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0xB0B0B0B0, 0x11111111, 0x11111111});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x11111111);
}

TEST_F(GcnTest, mad_mix_f32_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_MAD_MIX_F32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOpSelHi({0}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<float>(spirv, std::array{2.0f, 3.0f, 4.0f});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 10.0f);
}

TEST_F(GcnTest, mad_mix_f32_2) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_MAD_MIX_F32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOpSelHi({1,1,0}).SetOpSel({1,0,0}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<float>(spirv, std::array<u32,3>{
        std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(4.0f)}
    );

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 26.0f);
}

TEST_F(GcnTest, mad_mixlo_f16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_MAD_MIXLO_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOpSelHi({1,1,0}).SetOpSel({1,0,0}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<F16x2>(spirv, std::array<u32,3>{
        std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(4.0f)}
    );

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(26.0f), half(0.5f)}));
}

TEST_F(GcnTest, mad_mixhi_f16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_MAD_MIXHI_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOpSelHi({1,1,0}).SetOpSel({1,0,0}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<F16x2>(spirv, std::array<u32,3>{
        std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(4.0f)}
    );

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(44.0f), half(26.0f)}));
}

TEST_F(GcnTest, lshrrev_b16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_LSHRREV_B16, VOperand8::V0, SOperand9::V0, SOperand9::V1).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,2>{0xFFFFFFF2, 0x88881000});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xFFFF0400);
}

TEST_F(GcnTest, lshrrev_b16_2) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_LSHRREV_B16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetOpSel({0,0,0,1}).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,2>{0xFFFFFFF2, 0x88881000});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x0400FFF2);
}

TEST_F(GcnTest, lshrrev_b16_3) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_LSHRREV_B16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetOpSel({0,1,0,0}).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,2>{0xFFFFFFF2, 0x88881000});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xFFFF2222);
}

TEST_F(GcnTest, lshlrev_b16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_LSHLREV_B16, VOperand8::V0, SOperand9::V0, SOperand9::V1).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,2>{0xFFFFFFF3, 0x88888888});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xFFFF4440);
}

TEST_F(GcnTest, ashrrev_i16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ASHRREV_I16, VOperand8::V0, SOperand9::V0, SOperand9::V1).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,2>{0x1234FFF3, 0x88888888});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x1234F111);
}

TEST_F(GcnTest, pk_add_f16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f), half(2.0f)}, F16x2{half(3.0f), half(4.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(4.0f), half(6.0f)}));
}

TEST_F(GcnTest, pk_add_f16_2) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::Const0, SOperand9::ConstInv2Pi).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<u32>(spirv, 0U);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x00003118);
}

TEST_F(GcnTest, pk_add_f16_3) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::Const0, SOperand9::ConstInv2Pi).SetOpSel({0,1,1}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<u32>(spirv, 0U);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0);
}

TEST_F(GcnTest, pk_add_f16_4) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::Const0p5, SOperand9::Const0p5).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<u32>(spirv, 0U);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x3C00);
}

TEST_F(GcnTest, pk_add_f16_5) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::Const0, SOperand9::ConstInv2Pi).SetOpSelHi({0,0,0}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<u32>(spirv, 0U);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x31183118);
}

TEST_F(GcnTest, pk_add_f16_neg_lo) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetNeg({1,1,0}).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f), half(2.0f)}, F16x2{half(3.0f), half(4.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(-4.0f), half(6.0f)}));
}

TEST_F(GcnTest, pk_add_f16_neg_hi) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetNegHi({1,1,0}).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f), half(2.0f)}, F16x2{half(3.0f), half(4.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(4.0f), half(-6.0f)}));
}

TEST_F(GcnTest, pk_add_f16_op_sel_reversed) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetOpSel({1,1,1}).SetOpSelHi({0,0,0}).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f), half(2.0f)}, F16x2{half(3.0f), half(4.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(6.0f), half(4.0f)}));
}
