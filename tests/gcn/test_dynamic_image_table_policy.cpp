// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>

#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/dynamic_image_table_matcher.h"
#include "shader_recompiler/ir/passes/dynamic_image_table_policy.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/recompiler.h"

namespace Shader::Optimization {
namespace {

constexpr DynamicImageTablePattern ValidPattern() {
    return {
        .base_pointer_location = 18,
        .byte_offset = 14480,
        .byte_stride = 32,
        .is_r128 = true,
        .access = DynamicImageAccess::SampledExplicitLod,
        .index_upper_bound = std::nullopt,
        .supported_descriptor_window = 64,
        .supports_nonuniform_indexing = true,
        .supports_oob_guard = true,
    };
}

struct DynamicImageTableFixture {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    Gcn::Block cfg_block{};
    IR::Block* block{};
    IR::IREmitter* emitter{};

    DynamicImageTableFixture() {
        info.stage = Stage::Compute;
        info.l_stage = LogicalStage::Compute;
        block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        program.post_order_blocks.push_back(block);
        block->cfg_block = &cfg_block;
        emitter = new IR::IREmitter{*block};
        emitter->Prologue();
    }

    ~DynamicImageTableFixture() {
        delete emitter;
    }

    IR::Value MakeTablePointer(u32 low_offset = 2, u32 high_offset = 3) {
        auto& ir = *emitter;
        const auto root = ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0),
                                                ir.GetUserData(IR::ScalarReg::S1));
        return ir.CompositeConstruct(ir.ReadConst(root, ir.Imm32(low_offset)),
                                     ir.ReadConst(root, ir.Imm32(high_offset)));
    }

    IR::Inst* MakeSample(bool explicit_lod = true, u32 byte_offset = 14480, u32 byte_stride = 32,
                         u32 pointer_low_offset = 2, u32 pointer_high_offset = 3) {
        auto& ir = *emitter;
        const auto scaled_index = ir.IMul(ir.LaneId(), ir.Imm32(byte_stride));
        const auto byte_address = ir.IAdd(scaled_index, ir.Imm32(byte_offset));
        const auto dword_offset = ir.ShiftRightLogical(byte_address, ir.Imm32(2));
        const auto image =
            ir.ReadConst(MakeTablePointer(pointer_low_offset, pointer_high_offset), dword_offset);
        IR::TextureInstInfo texture_info{};
        texture_info.has_lod.Assign(explicit_lod);
        texture_info.is_r128.Assign(true);
        const auto access = ir.ImageSampleRaw(image, ir.Imm32(0), ir.Imm32(0), ir.Imm32(0),
                                              ir.Imm32(0), ir.Imm32(0), texture_info);
        ir.Reference(access);
        return access.InstRecursive();
    }
};

TEST(DynamicImageTablePolicy, PlansOnlyGuardedNonuniformExplicitLodSampling) {
    const auto plan = PlanDynamicImageTable(ValidPattern());
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->base_pointer_location, 18u);
    EXPECT_EQ(plan->offset_dw, 3620u);
    EXPECT_EQ(plan->stride_dw, 8u);
    EXPECT_EQ(plan->descriptor_count, 64u);
    EXPECT_EQ(plan->snapshot_span_dw, 508u);
    EXPECT_TRUE(plan->requires_oob_guard);

    for (const auto access :
         {DynamicImageAccess::SampledImplicitLod, DynamicImageAccess::StorageRead,
          DynamicImageAccess::StorageWrite, DynamicImageAccess::Atomic}) {
        auto rejected = ValidPattern();
        rejected.access = access;
        EXPECT_FALSE(PlanDynamicImageTable(rejected).has_value());
    }

    auto no_nonuniform = ValidPattern();
    no_nonuniform.supports_nonuniform_indexing = false;
    EXPECT_FALSE(PlanDynamicImageTable(no_nonuniform).has_value());

    auto no_guard = ValidPattern();
    no_guard.supports_oob_guard = false;
    EXPECT_FALSE(PlanDynamicImageTable(no_guard).has_value());
}

TEST(DynamicImageTablePolicy, TreatsTheWindowAsHostCapacityNotGuestLength) {
    auto bounded = ValidPattern();
    bounded.index_upper_bound = 5;
    bounded.supports_oob_guard = false;
    const auto within_window = PlanDynamicImageTable(bounded);
    ASSERT_TRUE(within_window.has_value());
    EXPECT_EQ(within_window->descriptor_count, 6u);
    EXPECT_FALSE(within_window->requires_oob_guard);

    bounded.index_upper_bound = 127;
    EXPECT_FALSE(PlanDynamicImageTable(bounded).has_value());
    bounded.supports_oob_guard = true;
    const auto clipped = PlanDynamicImageTable(bounded);
    ASSERT_TRUE(clipped.has_value());
    EXPECT_EQ(clipped->descriptor_count, 64u);
    EXPECT_TRUE(clipped->requires_oob_guard);

    bounded.index_upper_bound = std::numeric_limits<u32>::max();
    EXPECT_FALSE(PlanDynamicImageTable(bounded).has_value());
}

TEST(DynamicImageTablePolicy, FailsClosedOnMalformedOrOverflowingAddressShapes) {
    for (const u32 stride : {0u, 4u, 12u, 48u}) {
        auto malformed = ValidPattern();
        malformed.byte_stride = stride;
        EXPECT_FALSE(PlanDynamicImageTable(malformed).has_value()) << stride;
    }

    auto tightly_packed = ValidPattern();
    tightly_packed.byte_stride = 16;
    EXPECT_TRUE(PlanDynamicImageTable(tightly_packed).has_value());

    auto full_descriptor = tightly_packed;
    full_descriptor.is_r128 = false;
    EXPECT_FALSE(PlanDynamicImageTable(full_descriptor).has_value());
    full_descriptor.byte_stride = 32;
    EXPECT_TRUE(PlanDynamicImageTable(full_descriptor).has_value());

    auto unaligned_offset = ValidPattern();
    unaligned_offset.byte_offset++;
    EXPECT_FALSE(PlanDynamicImageTable(unaligned_offset).has_value());

    auto wrapping_address = ValidPattern();
    wrapping_address.byte_offset = 0xfffffff0u;
    wrapping_address.index_upper_bound = 1;
    wrapping_address.supported_descriptor_window = 2;
    EXPECT_FALSE(PlanDynamicImageTable(wrapping_address).has_value());

    auto exact_boundary = wrapping_address;
    exact_boundary.byte_offset = 0xffffffc0u;
    ASSERT_TRUE(PlanDynamicImageTable(exact_boundary).has_value());

    auto no_capacity = ValidPattern();
    no_capacity.supported_descriptor_window = 0;
    EXPECT_FALSE(PlanDynamicImageTable(no_capacity).has_value());

    auto enormous_window = tightly_packed;
    enormous_window.supported_descriptor_window = std::numeric_limits<u32>::max();
    EXPECT_FALSE(PlanDynamicImageTable(enormous_window).has_value());
}

TEST(DynamicImageTablePolicy, AppliesEveryDescriptorAndResourceLimitWithoutUnderflow) {
    EXPECT_EQ(DynamicImageDescriptorWindow(64, 0, 0, 256, 256, 256, 256), 64u);
    EXPECT_EQ(DynamicImageDescriptorWindow(64, 8, 9, 40, 256, 256, 256), 32u);
    EXPECT_EQ(DynamicImageDescriptorWindow(64, 8, 9, 256, 28, 256, 256), 20u);
    EXPECT_EQ(DynamicImageDescriptorWindow(64, 8, 9, 256, 256, 24, 256), 15u);
    EXPECT_EQ(DynamicImageDescriptorWindow(64, 8, 9, 256, 256, 256, 18), 10u);
    EXPECT_EQ(DynamicImageDescriptorWindow(64, 257, 9, 256, 256, 256, 256), 0u);
    EXPECT_EQ(DynamicImageDescriptorWindow(64, 8, 300, 256, 256, 256, 256), 0u);
}

TEST(DynamicImageTableMatcher, RecognizesOnlyTheAdjacentAffineSampleShape) {
    DynamicImageTableFixture fixture;
    const auto match = MatchDynamicImageTable(*fixture.MakeSample());
    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(match->byte_offset, 14480u);
    EXPECT_EQ(match->byte_stride, 32u);
    EXPECT_EQ(match->access, DynamicImageAccess::SampledExplicitLod);
    EXPECT_TRUE(match->is_r128);
    EXPECT_FALSE(match->index_upper_bound.has_value());
    ASSERT_NE(match->image_handle, nullptr);
    ASSERT_NE(match->pointer_lo, nullptr);
    ASSERT_NE(match->pointer_hi, nullptr);

    DynamicImageTableFixture nonadjacent;
    EXPECT_FALSE(
        MatchDynamicImageTable(*nonadjacent.MakeSample(true, 14480, 32, 2, 4)).has_value());

    DynamicImageTableFixture wrapped_pair;
    EXPECT_FALSE(MatchDynamicImageTable(
                     *wrapped_pair.MakeSample(true, 14480, 32, std::numeric_limits<u32>::max(), 0))
                     .has_value());
}

TEST(DynamicImageTableMatcher, SeparatesRecognitionFromFailClosedAdmission) {
    DynamicImageTableFixture implicit;
    const auto match = MatchDynamicImageTable(*implicit.MakeSample(false));
    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(match->access, DynamicImageAccess::SampledImplicitLod);
    EXPECT_FALSE(PlanDynamicImageTable(match->Pattern(18, 64, true, true)).has_value());

    DynamicImageTableFixture unaligned;
    const auto unaligned_match = MatchDynamicImageTable(*unaligned.MakeSample(true, 14481));
    ASSERT_TRUE(unaligned_match.has_value());
    EXPECT_FALSE(PlanDynamicImageTable(unaligned_match->Pattern(18, 64, true, true)).has_value());

    DynamicImageTableFixture bad_stride;
    const auto stride_match = MatchDynamicImageTable(*bad_stride.MakeSample(true, 14480, 48));
    ASSERT_TRUE(stride_match.has_value());
    EXPECT_FALSE(PlanDynamicImageTable(stride_match->Pattern(18, 64, true, true)).has_value());
}

TEST(DynamicImageTableMatcher, DeclinesRawFormsWithDifferentBackendSemantics) {
    const auto rejected = [](auto mutate_info) {
        DynamicImageTableFixture fixture;
        IR::Inst* const access = fixture.MakeSample();
        auto info = access->Flags<IR::TextureInstInfo>();
        mutate_info(info);
        access->SetFlags(info);
        return !MatchDynamicImageTable(*access).has_value();
    };

    EXPECT_TRUE(rejected([](auto& info) { info.is_depth.Assign(true); }));
    EXPECT_TRUE(rejected([](auto& info) { info.is_gather.Assign(true); }));
    EXPECT_TRUE(rejected([](auto& info) { info.has_derivatives.Assign(true); }));
    EXPECT_TRUE(rejected([](auto& info) { info.has_lod_clamp.Assign(true); }));
    EXPECT_TRUE(rejected([](auto& info) { info.is_unnormalized.Assign(true); }));
    EXPECT_FALSE(rejected([](auto& info) { info.has_offset.Assign(true); }));
    EXPECT_FALSE(rejected([](auto& info) { info.is_array.Assign(true); }));
    EXPECT_FALSE(rejected([](auto& info) {
        info.has_lod.Assign(false);
        info.force_level0.Assign(true);
    }));
}

TEST(DynamicImageTableMatcher, PointerPhiTraversalIsCycleSafeAndUnique) {
    DynamicImageTableFixture fixture;
    auto& ir = *fixture.emitter;
    const auto first = ir.GetUserData(IR::ScalarReg::S2);
    const auto second = ir.GetUserData(IR::ScalarReg::S3);

    IR::Inst* const same_source_phi =
        &*fixture.block->PrependNewInst(fixture.block->begin(), IR::Opcode::Phi);
    same_source_phi->SetFlags(IR::Type::U32);
    same_source_phi->AddPhiOperand(fixture.block, first);
    same_source_phi->AddPhiOperand(fixture.block, first);
    EXPECT_EQ(Detail::FindUniquePointerSource(*same_source_phi), first.InstRecursive());

    IR::Inst* const ambiguous_phi =
        &*fixture.block->PrependNewInst(fixture.block->begin(), IR::Opcode::Phi);
    ambiguous_phi->SetFlags(IR::Type::U32);
    ambiguous_phi->AddPhiOperand(fixture.block, first);
    ambiguous_phi->AddPhiOperand(fixture.block, second);
    EXPECT_EQ(Detail::FindUniquePointerSource(*ambiguous_phi), nullptr);

    IR::Inst* const source_less_cycle =
        &*fixture.block->PrependNewInst(fixture.block->begin(), IR::Opcode::Phi);
    source_less_cycle->SetFlags(IR::Type::U32);
    source_less_cycle->AddPhiOperand(fixture.block, IR::U32{source_less_cycle});
    EXPECT_EQ(Detail::FindUniquePointerSource(*source_less_cycle), nullptr);
}

TEST(DynamicImageTableMatcher, DerivesOnlySoundImmediateIndexBounds) {
    DynamicImageTableFixture and_fixture;
    auto& and_ir = *and_fixture.emitter;
    const auto bounded_index = and_ir.BitwiseAnd(and_ir.LaneId(), and_ir.Imm32(5));
    EXPECT_EQ(DynamicImageIndexUpperBound(bounded_index), 5u);

    DynamicImageTableFixture min_fixture;
    auto& min_ir = *min_fixture.emitter;
    const auto min_index = min_ir.UMin(min_ir.LaneId(), min_ir.Imm32(7));
    EXPECT_EQ(DynamicImageIndexUpperBound(min_index), 7u);

    const auto reverse_min_index = min_ir.UMin(min_ir.Imm32(9), min_ir.LaneId());
    EXPECT_EQ(DynamicImageIndexUpperBound(reverse_min_index), 9u);

    const auto signed_min_index = min_ir.SMin(min_ir.LaneId(), min_ir.Imm32(11));
    EXPECT_FALSE(DynamicImageIndexUpperBound(signed_min_index).has_value());

    DynamicImageTableFixture unbounded_fixture;
    EXPECT_FALSE(DynamicImageIndexUpperBound(unbounded_fixture.emitter->LaneId()).has_value());
}

} // namespace
} // namespace Shader::Optimization
