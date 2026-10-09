// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <functional>
#include <initializer_list>
#include <limits>

#include <gtest/gtest.h>
#include "video_core/renderer_vulkan/descriptor_layout_policy.h"

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

    IR::Inst* MakePointerSample(bool commute = false, u32 byte_offset = 14480U,
                                u32 read_offset_dw = 0U) {
        auto& ir = *emitter;
        const auto base = MakeTablePointer();
        const IR::U32 base_low{base.InstRecursive()->Arg(0)};
        const IR::U32 base_high{base.InstRecursive()->Arg(1)};
        const IR::U32 offset{ir.IAdd(ir.IMul(ir.LaneId(), ir.Imm32(32U)), ir.Imm32(byte_offset))};
        const IR::U32 low{commute ? ir.IAdd(base_low, offset) : ir.IAdd(offset, base_low)};
        const IR::U32 carry{ir.Select(ir.ILessThan(low, commute ? base_low : offset, false),
                                      ir.Imm32(1U), ir.Imm32(0U))};
        const auto high = commute ? ir.IAdd(carry, base_high) : ir.IAdd(base_high, carry);
        const auto image = ir.ReadConst(ir.CompositeConstruct(low, high), ir.Imm32(read_offset_dw));
        IR::TextureInstInfo texture_info{};
        texture_info.has_lod.Assign(true);
        const auto access = ir.ImageSampleRaw(image, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U),
                                              ir.Imm32(0U), ir.Imm32(0U), texture_info);
        ir.Reference(access);
        return access.InstRecursive();
    }

    IR::Inst* MakeGpuDerivedDescriptorTableSample() {
        auto& ir = *emitter;
        const auto root =
            ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0),
                                  ir.GetUserData(IR::ScalarReg::S1));
        const auto base_low = ir.ReadConst(root, ir.Imm32(2U));
        const auto base_high = ir.ReadConst(root, ir.Imm32(3U));
        const IR::U32 gpu_value{ir.LoadShared(32, false, ir.Imm32(0U), false)};
        const IR::U32 runtime_low{ir.ReadFirstLane(ir.IAdd(base_low, gpu_value))};
        const auto runtime_pointer = ir.CompositeConstruct(runtime_low, base_high);
        const auto table = ir.CompositeConstruct(
            ir.ReadConst(runtime_pointer, ir.Imm32(0U)),
            ir.ReadConst(runtime_pointer, ir.Imm32(1U)));
        const auto scaled_index = ir.IMul(ir.LaneId(), ir.Imm32(32U));
        const auto byte_address = ir.IAdd(scaled_index, ir.Imm32(14480U));
        const auto dword_offset = ir.ShiftRightLogical(byte_address, ir.Imm32(2U));
        const auto image = ir.ReadConst(table, dword_offset);
        IR::TextureInstInfo texture_info{};
        texture_info.has_lod.Assign(true);
        texture_info.is_r128.Assign(true);
        const auto access = ir.ImageSampleRaw(image, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U),
                                              ir.Imm32(0U), ir.Imm32(0U), texture_info);
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

TEST(DynamicImageTableMatcher, PointerFormPreservesTheExactCarryAndBothBaseHalves) {
    for (const bool commute : {false, true}) {
        DynamicImageTableFixture fixture;
        const auto match = MatchDynamicImageTable(*fixture.MakePointerSample(commute, 14480U, 2U));
        ASSERT_TRUE(match.has_value());
        EXPECT_EQ(match->byte_offset, 14488U);
        EXPECT_EQ(match->byte_stride, 32U);
        ASSERT_TRUE(PlanDynamicImageTable(match->Pattern(18U, 64U, true, true)));
        const auto* const pointer = match->image_handle->Arg(0).InstRecursive();
        for (const u64 base : {0x1234000000ULL, 0x12FFFFFFF0ULL, 0x00FFFFFFF0ULL}) {
            for (const u32 index : {0U, 1U, 31U, 63U}) {
                std::function<u32(IR::Value)> evaluate = [&](IR::Value value) -> u32 {
                    if (value.IsImmediate()) {
                        return value.U32();
                    }
                    auto* const inst = value.InstRecursive();
                    if (inst == match->pointer_lo) {
                        return static_cast<u32>(base);
                    }
                    if (inst == match->pointer_hi) {
                        return static_cast<u32>(base >> 32U);
                    }
                    if (value == match->index) {
                        return index;
                    }
                    switch (inst->GetOpcode()) {
                    case IR::Opcode::IAdd32:
                        return evaluate(inst->Arg(0)) + evaluate(inst->Arg(1));
                    case IR::Opcode::IMul32:
                        return evaluate(inst->Arg(0)) * evaluate(inst->Arg(1));
                    case IR::Opcode::ULessThan32:
                        return evaluate(inst->Arg(0)) < evaluate(inst->Arg(1));
                    case IR::Opcode::SelectU32:
                        return evaluate(inst->Arg(0)) ? evaluate(inst->Arg(1))
                                                      : evaluate(inst->Arg(2));
                    default:
                        ADD_FAILURE() << "Unexpected scalar in synthetic pointer";
                        return 0U;
                    }
                };
                const u64 actual_pointer =
                    u64{evaluate(pointer->Arg(0))} | (u64{evaluate(pointer->Arg(1))} << 32U);
                const u64 actual_address = actual_pointer + 2U * sizeof(u32);
                EXPECT_EQ(actual_address, base + match->byte_offset + index * match->byte_stride)
                    << "base=" << base << " index=" << index;
            }
        }
    }
}

TEST(DynamicImageTableMatcher, DeclinesGpuDerivedDescriptorTablePointer) {
    DynamicImageTableFixture fixture;

    // The descriptor-table address is read from a pointer whose low half depends on a GPU-only
    // value. It must remain a runtime read; the host SRT walker cannot safely dereference it.
    EXPECT_FALSE(
        MatchDynamicImageTable(*fixture.MakeGpuDerivedDescriptorTableSample()).has_value());
}

TEST(DynamicImageTableMatcher, RejectsOrdinaryFloatVectorPayloadAsDescriptorPointer) {
    DynamicImageTableFixture fixture;
    auto& ir = *fixture.emitter;
    const auto root =
        ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0), ir.GetUserData(IR::ScalarReg::S1));

    // These adjacent words are ordinary payload (for example, a float2) read from the same
    // host-readable constant source. The later U32 re-composition is intentionally malformed
    // provenance: the matcher currently sees only the final shape and may mistake the payload
    // for a descriptor-table pointer.
    const auto payload_low = ir.ReadConst(root, ir.Imm32(64U));
    const auto payload_high = ir.ReadConst(root, ir.Imm32(65U));
    auto* const low_identity = &*fixture.block->PrependNewInst(
        fixture.block->begin(), IR::Opcode::Identity, {IR::Value{payload_low}});
    auto* const high_phi = &*fixture.block->PrependNewInst(fixture.block->begin(), IR::Opcode::Phi);
    high_phi->SetFlags(IR::Type::U32);
    high_phi->AddPhiOperand(fixture.block, payload_high);
    high_phi->AddPhiOperand(fixture.block, payload_high);
    const auto ordinary_float2 = ir.CompositeConstruct(
        ir.BitCast<IR::F32>(IR::U32{IR::Value{low_identity}}),
        ir.BitCast<IR::F32>(IR::U32{IR::Value{high_phi}}));
    ir.Reference(ordinary_float2);

    const auto candidate_pointer = ir.CompositeConstruct(payload_low, payload_high);
    const auto scaled_index = ir.IMul(ir.LaneId(), ir.Imm32(32U));
    const auto byte_address = ir.IAdd(scaled_index, ir.Imm32(14480U));
    const auto dword_offset = ir.ShiftRightLogical(byte_address, ir.Imm32(2U));
    const auto image = ir.ReadConst(candidate_pointer, dword_offset);
    IR::TextureInstInfo texture_info{};
    texture_info.has_lod.Assign(true);
    texture_info.is_r128.Assign(true);
    const auto access = ir.ImageSampleRaw(image, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U),
                                          ir.Imm32(0U), ir.Imm32(0U), texture_info);
    ir.Reference(access);

    // The source-role provenance guard rejects this ambiguous pair while leaving pointer-only
    // ReadConst pairs eligible for flattening.
    EXPECT_TRUE(Detail::HasFloatingPointUse(*payload_low.InstRecursive()));
    EXPECT_TRUE(Detail::HasFloatingPointUse(*payload_high.InstRecursive()));
    EXPECT_FALSE(MatchDynamicImageTable(*access.InstRecursive()).has_value());
}

TEST(DynamicImageTableMatcher, DeadFloatingPointConsumerDoesNotChangeTablePointerAdmission) {
    for (const bool forwarded : {false, true}) {
        SCOPED_TRACE(forwarded);
        DynamicImageTableFixture fixture;
        auto& ir = *fixture.emitter;
        auto* const access = fixture.MakeSample();
        auto* const pointer = access->Arg(0).InstRecursive()->Arg(0).InstRecursive();
        const IR::U32 low{pointer->Arg(0)};
        const IR::U32 high{pointer->Arg(1)};
        const IR::U32 low_value{forwarded ? IR::Value{ir.IAdd(low, ir.Imm32(0U))}
                                         : IR::Value{low}};
        const IR::U32 high_value{forwarded ? ir.Select(ir.Imm1(true), high, high)
                                          : IR::Value{high}};
        const auto low_float = ir.BitCast<IR::F32>(low_value);
        const auto high_float = ir.BitCast<IR::F32>(high_value);
        ASSERT_FALSE(low_float.InstRecursive()->HasUses());
        ASSERT_FALSE(high_float.InstRecursive()->HasUses());
        ASSERT_FALSE(low_float.InstRecursive()->MayHaveSideEffects());
        ASSERT_FALSE(high_float.InstRecursive()->MayHaveSideEffects());

        EXPECT_FALSE(Detail::HasFloatingPointUse(*low.InstRecursive()));
        EXPECT_FALSE(Detail::HasFloatingPointUse(*high.InstRecursive()));
        EXPECT_TRUE(MatchDynamicImageTable(*access).has_value());

        // This is only a liveness correction. The existing conservative policy must still
        // reject a floating interpretation as soon as it becomes observable.
        ir.Reference(high_float);
        EXPECT_TRUE(Detail::HasFloatingPointUse(*high.InstRecursive()));
        EXPECT_FALSE(MatchDynamicImageTable(*access).has_value());
    }
}

TEST(DynamicImageTableMatcher, RejectsFloatingPointUseThroughAllowedScalarForwarding) {
    DynamicImageTableFixture fixture;
    auto& ir = *fixture.emitter;
    const auto root =
        ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0), ir.GetUserData(IR::ScalarReg::S1));
    const auto low = ir.ReadConst(root, ir.Imm32(2U));
    const auto high = ir.ReadConst(root, ir.Imm32(3U));
    const auto low_forwarded = ir.IAdd(low, ir.Imm32(0U));
    const auto high_forwarded = ir.Select(ir.Imm1(true), high, high);
    ir.Reference(ir.BitCast<IR::F32>(IR::U32{low_forwarded}));
    ir.Reference(ir.BitCast<IR::F32>(IR::U32{high_forwarded}));

    const auto table = ir.CompositeConstruct(low, high);
    const auto index = ir.IMul(ir.LaneId(), ir.Imm32(32U));
    const auto address = ir.IAdd(index, ir.Imm32(14480U));
    const auto image = ir.ReadConst(table, ir.ShiftRightLogical(address, ir.Imm32(2U)));
    IR::TextureInstInfo texture_info{};
    texture_info.has_lod.Assign(true);
    texture_info.is_r128.Assign(true);
    const auto access = ir.ImageSampleRaw(image, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U),
                                          ir.Imm32(0U), ir.Imm32(0U), texture_info);
    ir.Reference(access);

    EXPECT_TRUE(Detail::HasFloatingPointUse(*low.InstRecursive()));
    EXPECT_TRUE(Detail::HasFloatingPointUse(*high.InstRecursive()));
    EXPECT_FALSE(MatchDynamicImageTable(*access.InstRecursive()).has_value());
}

TEST(DynamicImageTableMatcher, RejectsFloatingPointUseThroughUnsignedMinForwarding) {
    DynamicImageTableFixture fixture;
    auto& ir = *fixture.emitter;
    auto* const access = fixture.MakeSample();
    auto* const pointer = access->Arg(0).InstRecursive()->Arg(0).InstRecursive();
    const IR::U32 low{pointer->Arg(0)};

    // UMin32 is an integer result, but it still carries bits from either operand into a later
    // consumer. The host table matcher must not treat this path as pointer-only when the result
    // is reinterpreted as floating-point data.
    const auto bounded_low = ir.UMin(low, ir.Imm32(0x7fffffffu));
    ir.Reference(ir.BitCast<IR::F32>(bounded_low));

    EXPECT_TRUE(Detail::HasFloatingPointUse(*low.InstRecursive()));
    EXPECT_FALSE(MatchDynamicImageTable(*access).has_value());
}

TEST(DynamicImageTableMatcher, RejectsFloatingPointUseThroughUserDataPointer) {
    DynamicImageTableFixture fixture;
    auto& ir = *fixture.emitter;
    const auto low = ir.GetUserData(IR::ScalarReg::S0);
    const auto high = ir.GetUserData(IR::ScalarReg::S1);
    ir.Reference(ir.BitCast<IR::F32>(low));
    ir.Reference(ir.BitCast<IR::F32>(high));
    const auto pointer = ir.CompositeConstruct(low, high);
    const auto index = ir.IMul(ir.LaneId(), ir.Imm32(32U));
    const auto address = ir.IAdd(index, ir.Imm32(14480U));
    const auto image = ir.ReadConst(pointer, ir.ShiftRightLogical(address, ir.Imm32(2U)));
    IR::TextureInstInfo texture_info{};
    texture_info.has_lod.Assign(true);
    texture_info.is_r128.Assign(true);
    const auto access = ir.ImageSampleRaw(image, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U),
                                          ir.Imm32(0U), ir.Imm32(0U), texture_info);
    ir.Reference(access);

    EXPECT_TRUE(Detail::HasFloatingPointUse(*low.InstRecursive()));
    EXPECT_TRUE(Detail::HasFloatingPointUse(*high.InstRecursive()));
    EXPECT_FALSE(MatchDynamicImageTable(*access.InstRecursive()).has_value());
}

TEST(DynamicImageTableMatcher, RejectsFloatingPointUseThroughCompositeExtraction) {
    DynamicImageTableFixture fixture;
    auto& ir = *fixture.emitter;
    const auto root =
        ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0), ir.GetUserData(IR::ScalarReg::S1));
    const auto low = ir.ReadConst(root, ir.Imm32(2U));
    const auto high = ir.ReadConst(root, ir.Imm32(3U));
    const auto pointer = ir.CompositeConstruct(low, high);
    const auto extracted_low = IR::U32{ir.CompositeExtract(pointer, 0U)};
    const auto extracted_high = IR::U32{ir.CompositeExtract(pointer, 1U)};
    ir.Reference(ir.BitCast<IR::F32>(extracted_low));
    ir.Reference(ir.BitCast<IR::F32>(extracted_high));
    const auto index = ir.IMul(ir.LaneId(), ir.Imm32(32U));
    const auto address = ir.IAdd(index, ir.Imm32(14480U));
    const auto image = ir.ReadConst(pointer, ir.ShiftRightLogical(address, ir.Imm32(2U)));
    IR::TextureInstInfo texture_info{};
    texture_info.has_lod.Assign(true);
    texture_info.is_r128.Assign(true);
    const auto access = ir.ImageSampleRaw(image, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U),
                                          ir.Imm32(0U), ir.Imm32(0U), texture_info);
    ir.Reference(access);

    EXPECT_TRUE(Detail::HasFloatingPointUse(*low.InstRecursive()));
    EXPECT_TRUE(Detail::HasFloatingPointUse(*high.InstRecursive()));
    EXPECT_FALSE(MatchDynamicImageTable(*access.InstRecursive()).has_value());
}

TEST(DynamicImageTableMatcher, RejectsFloatingPointUseThroughPackUnpack) {
    DynamicImageTableFixture fixture;
    auto& ir = *fixture.emitter;
    const auto root =
        ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0), ir.GetUserData(IR::ScalarReg::S1));
    const auto low = ir.ReadConst(root, ir.Imm32(2U));
    const auto high = ir.ReadConst(root, ir.Imm32(3U));

    // Packing and unpacking is lossless value forwarding. Both pointer halves must remain marked
    // ambiguous when the unpacked words are subsequently interpreted as floating-point data.
    const auto packed = ir.PackUint2x32(ir.CompositeConstruct(low, high));
    const auto unpacked = ir.UnpackUint2x32(packed);
    const IR::U32 unpacked_low{ir.CompositeExtract(unpacked, 0U)};
    const IR::U32 unpacked_high{ir.CompositeExtract(unpacked, 1U)};
    ir.Reference(ir.BitCast<IR::F32>(unpacked_low));
    ir.Reference(ir.BitCast<IR::F32>(unpacked_high));

    const auto table = ir.CompositeConstruct(low, high);
    const auto index = ir.IMul(ir.LaneId(), ir.Imm32(32U));
    const auto address = ir.IAdd(index, ir.Imm32(14480U));
    const auto image = ir.ReadConst(table, ir.ShiftRightLogical(address, ir.Imm32(2U)));
    IR::TextureInstInfo texture_info{};
    texture_info.has_lod.Assign(true);
    texture_info.is_r128.Assign(true);
    const auto access = ir.ImageSampleRaw(image, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U),
                                          ir.Imm32(0U), ir.Imm32(0U), texture_info);
    ir.Reference(access);

    EXPECT_TRUE(Detail::HasFloatingPointUse(*low.InstRecursive()));
    EXPECT_TRUE(Detail::HasFloatingPointUse(*high.InstRecursive()));
    EXPECT_FALSE(MatchDynamicImageTable(*access.InstRecursive()).has_value());
}

TEST(DynamicImageTableMatcher, RejectsFloatingPointUseThroughIntegerWidthConversions) {
    for (const u32 width : {8U, 16U}) {
        DynamicImageTableFixture fixture;
        auto& ir = *fixture.emitter;
        auto* const access = fixture.MakeSample();
        auto* const pointer = access->Arg(0).InstRecursive()->Arg(0).InstRecursive();
        const IR::U32 low{pointer->Arg(0)};

        // Both narrowing/zero-extension and the signed extension spelling carry bits from the
        // original U32 word. A later float endpoint must therefore make the source ambiguous.
        const auto narrowed = ir.UConvert(width, low);
        const IR::U32 round_trip{ir.UConvert(32, narrowed)};
        const IR::U32 signed_round_trip{ir.SConvert(32, narrowed)};
        ir.Reference(ir.BitCast<IR::F32>(round_trip));
        ir.Reference(ir.BitCast<IR::F32>(signed_round_trip));

        EXPECT_TRUE(Detail::HasFloatingPointUse(*low.InstRecursive())) << width;
        EXPECT_FALSE(MatchDynamicImageTable(*access).has_value()) << width;
    }

    DynamicImageTableFixture wide_fixture;
    auto& wide_ir = *wide_fixture.emitter;
    auto* const access = wide_fixture.MakeSample();
    auto* const pointer = access->Arg(0).InstRecursive()->Arg(0).InstRecursive();
    const IR::U32 low{pointer->Arg(0)};
    const IR::U64 widened{wide_ir.UConvert(64, low)};
    const auto unpacked = wide_ir.UnpackUint2x32(widened);
    const IR::U32 recovered{wide_ir.CompositeExtract(unpacked, 0U)};
    wide_ir.Reference(wide_ir.BitCast<IR::F32>(recovered));

    EXPECT_TRUE(Detail::HasFloatingPointUse(*low.InstRecursive()));
    EXPECT_FALSE(MatchDynamicImageTable(*access).has_value());
}

TEST(DynamicImageTableMatcher, PointerHalvesUsedAsFloatControlRemainEligible) {
    DynamicImageTableFixture fixture;
    auto& ir = *fixture.emitter;
    const auto root =
        ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0), ir.GetUserData(IR::ScalarReg::S1));
    const auto low = ir.ReadConst(root, ir.Imm32(2U));
    const auto high = ir.ReadConst(root, ir.Imm32(3U));

    // The pointer half only controls which floating-point value is selected. It is not itself
    // interpreted as floating-point data, so this control dependency must not disqualify it.
    const auto float_select =
        ir.Select(ir.ILessThan(low, ir.Imm32(1U), false), ir.Imm32(1.0f), ir.Imm32(2.0f));
    ir.Reference(float_select);

    const auto table = ir.CompositeConstruct(low, high);
    const auto index = ir.IMul(ir.LaneId(), ir.Imm32(32U));
    const auto address = ir.IAdd(index, ir.Imm32(14480U));
    const auto image = ir.ReadConst(table, ir.ShiftRightLogical(address, ir.Imm32(2U)));
    IR::TextureInstInfo texture_info{};
    texture_info.has_lod.Assign(true);
    texture_info.is_r128.Assign(true);
    const auto access = ir.ImageSampleRaw(image, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U),
                                          ir.Imm32(0U), ir.Imm32(0U), texture_info);
    ir.Reference(access);

    EXPECT_FALSE(Detail::HasFloatingPointUse(*low.InstRecursive()));
    EXPECT_FALSE(Detail::HasFloatingPointUse(*high.InstRecursive()));
    EXPECT_TRUE(MatchDynamicImageTable(*access.InstRecursive()).has_value());
}

TEST(DynamicImageTableMatcher, HighFanoutPhiFailsClosedBeforePendingQueueGrowth) {
    DynamicImageTableFixture fixture;
    auto& ir = *fixture.emitter;
    const auto source = ir.GetUserData(IR::ScalarReg::S0);
    auto* const phi = &*fixture.block->PrependNewInst(fixture.block->begin(), IR::Opcode::Phi);
    phi->SetFlags(IR::Type::U32);
    // Repeating one source as many Phi operands keeps the reachable graph tiny while exercising
    // the worklist's duplicate-enqueue bound. A conservative provenance result is required when
    // the pending queue itself would exceed its node budget.
    for (size_t operand = 0; operand < 300; ++operand) {
        phi->AddPhiOperand(fixture.block, source);
    }
    ir.Reference(ir.IAdd(IR::U32{phi}, ir.Imm32(0U)));

    EXPECT_TRUE(Detail::HasFloatingPointUse(*source.InstRecursive()));
}

TEST(DynamicImageTableMatcher, PointerFormRejectsMissingOrUnrelatedCarryAndWrongHighHalf) {
    enum class Mutation { MissingCarry, DifferentSum, ReversedComparison, NonBitCarry, WrongHigh };
    for (const auto mutation :
         {Mutation::MissingCarry, Mutation::DifferentSum, Mutation::ReversedComparison,
          Mutation::NonBitCarry, Mutation::WrongHigh}) {
        DynamicImageTableFixture fixture;
        auto& ir = *fixture.emitter;
        auto* const access = fixture.MakePointerSample();
        auto* const pointer = access->Arg(0).InstRecursive()->Arg(0).InstRecursive();
        auto* const high = pointer->Arg(1).InstRecursive();
        auto* const carry = high->Arg(1).InstRecursive();
        auto* const comparison = carry->Arg(0).InstRecursive();
        switch (mutation) {
        case Mutation::MissingCarry:
            pointer->SetArg(1, high->Arg(0));
            break;
        case Mutation::DifferentSum:
            comparison->SetArg(0, ir.IAdd(IR::U32{pointer->Arg(0)}, ir.Imm32(1U)));
            break;
        case Mutation::ReversedComparison: {
            const auto lhs = comparison->Arg(0);
            comparison->SetArg(0, comparison->Arg(1));
            comparison->SetArg(1, lhs);
            break;
        }
        case Mutation::NonBitCarry:
            carry->SetArg(1, ir.Imm32(2U));
            break;
        case Mutation::WrongHigh:
            high->SetArg(0, ir.GetUserData(IR::ScalarReg::S7));
            break;
        }
        EXPECT_FALSE(MatchDynamicImageTable(*access).has_value()) << static_cast<int>(mutation);
    }
}

TEST(DynamicImageTableMatcher, PointerFormDoesNotHideOffsetArithmeticWrap) {
    DynamicImageTableFixture wraps_within_window;
    const auto match =
        MatchDynamicImageTable(*wraps_within_window.MakePointerSample(false, 0xFFFFFFF0U));
    ASSERT_TRUE(match.has_value());
    EXPECT_FALSE(PlanDynamicImageTable(match->Pattern(18U, 64U, true, true)));

    DynamicImageTableFixture constant_offset_overflow;
    EXPECT_FALSE(MatchDynamicImageTable(*constant_offset_overflow.MakePointerSample(
        false, 14480U, std::numeric_limits<u32>::max())));
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

IR::U32 UndefPointer(DynamicImageTableFixture& fixture) {
    return IR::U32{&*fixture.block->PrependNewInst(fixture.block->begin(), IR::Opcode::UndefU32)};
}

IR::Inst* PointerPhi(DynamicImageTableFixture& fixture, std::initializer_list<IR::Value> operands) {
    auto* const phi = &*fixture.block->PrependNewInst(fixture.block->begin(), IR::Opcode::Phi);
    phi->SetFlags(IR::Type::U32);
    for (const auto& operand : operands) {
        phi->AddPhiOperand(fixture.block, operand);
    }
    return phi;
}

TEST(DynamicImageTableMatcher, RejectsFloatingPointUseThroughCompositeInsertShuffleAndPhi) {
    DynamicImageTableFixture fixture;
    auto& ir = *fixture.emitter;
    auto* const access = fixture.MakeSample();
    auto* const pointer = access->Arg(0).InstRecursive()->Arg(0).InstRecursive();
    const IR::U32 low{pointer->Arg(0)};
    const IR::U32 high{pointer->Arg(1)};
    // CompositeShuffle's 2-result form selects from the four-component input vectors in this IR.
    const auto pair = ir.CompositeConstruct(low, high, ir.Imm32(0U), ir.Imm32(0U));

    const auto inserted = ir.CompositeInsert(pair, low, 0U);
    const IR::U32 inserted_low{ir.CompositeExtract(inserted, 0U)};
    ir.Reference(ir.BitCast<IR::F32>(inserted_low));

    const auto shuffled = ir.CompositeShuffle(pair, pair, 1U, 0U);
    const IR::U32 shuffled_low{ir.CompositeExtract(shuffled, 0U)};
    ir.Reference(ir.BitCast<IR::F32>(shuffled_low));

    auto* const phi = PointerPhi(fixture, {low, low});
    ir.Reference(ir.BitCast<IR::F32>(IR::U32{phi}));

    EXPECT_TRUE(Detail::HasFloatingPointUse(*low.InstRecursive()));
    EXPECT_TRUE(Detail::HasFloatingPointUse(*high.InstRecursive()));
    EXPECT_FALSE(MatchDynamicImageTable(*access).has_value());
}

TEST(DynamicImageTableMatcher, PointerPhiRefinesNestedUndefToOneDefinedSource) {
    DynamicImageTableFixture fixture;
    const auto source = fixture.emitter->GetUserData(IR::ScalarReg::S2);
    const auto undef = UndefPointer(fixture);
    auto* const inner = PointerPhi(fixture, {source, undef});
    auto* const outer = PointerPhi(fixture, {source, IR::Value{inner}});

    EXPECT_EQ(Detail::FindUniquePointerSource(*outer), source.InstRecursive());
}

TEST(DynamicImageTableMatcher, PointerPhiRefinesUndefInAnAnchoredLoop) {
    DynamicImageTableFixture fixture;
    const auto source = fixture.emitter->GetUserData(IR::ScalarReg::S2);
    auto* const phi = PointerPhi(fixture, {source, UndefPointer(fixture)});
    phi->AddPhiOperand(fixture.block, IR::Value{phi});

    EXPECT_EQ(Detail::FindUniquePointerSource(*phi), source.InstRecursive());
}

TEST(DynamicImageTableMatcher, PointerPhiAllUndefHasNoDefinedAnchor) {
    DynamicImageTableFixture fixture;
    const auto first = UndefPointer(fixture);
    const auto second = UndefPointer(fixture);
    auto* const phi = PointerPhi(fixture, {first, second});

    EXPECT_EQ(Detail::FindUniquePointerSource(*first.InstRecursive()), nullptr);
    EXPECT_EQ(Detail::FindUniquePointerSource(*phi), nullptr);
}

TEST(DynamicImageTableMatcher, PointerPhiUndefAndAnchorlessCycleRemainRefused) {
    DynamicImageTableFixture fixture;
    auto* const first = PointerPhi(fixture, {UndefPointer(fixture)});
    auto* const second = PointerPhi(fixture, {IR::Value{first}});
    first->AddPhiOperand(fixture.block, IR::Value{second});

    EXPECT_EQ(Detail::FindUniquePointerSource(*first), nullptr);
    EXPECT_EQ(Detail::FindUniquePointerSource(*second), nullptr);
}

TEST(DynamicImageTableMatcher, PointerPhiUndefDoesNotHideDistinctDefinedSources) {
    DynamicImageTableFixture fixture;
    const auto first = fixture.emitter->GetUserData(IR::ScalarReg::S2);
    const auto second = fixture.emitter->GetUserData(IR::ScalarReg::S3);
    auto* const inner = PointerPhi(fixture, {first, UndefPointer(fixture)});
    auto* const outer = PointerPhi(fixture, {second, IR::Value{inner}});

    EXPECT_EQ(Detail::FindUniquePointerSource(*outer), nullptr);
}

TEST(DynamicImageTableMatcher, PointerPhiUndefDoesNotHideAnUnsupportedExpression) {
    DynamicImageTableFixture fixture;
    const auto source = fixture.emitter->GetUserData(IR::ScalarReg::S2);
    const auto adjusted = fixture.emitter->IAdd(source, fixture.emitter->Imm32(4U));
    auto* const phi = PointerPhi(fixture, {source, UndefPointer(fixture), adjusted});

    EXPECT_EQ(Detail::FindUniquePointerSource(*phi), nullptr);
}

TEST(DynamicImageTableMatcher, PointerPhiZeroAndOtherImmediatesAreNotUndefined) {
    for (const u32 immediate : {0U, 1U, 0xFFFFFFFFU}) {
        DynamicImageTableFixture fixture;
        const auto source = fixture.emitter->GetUserData(IR::ScalarReg::S2);
        auto* const phi =
            PointerPhi(fixture, {source, UndefPointer(fixture), fixture.emitter->Imm32(immediate)});

        EXPECT_EQ(Detail::FindUniquePointerSource(*phi), nullptr) << immediate;
    }
}

TEST(DynamicImageTableMatcher, DeepPointerPhiTraversalFailsClosedAtTheWorklistBound) {
    DynamicImageTableFixture fixture;
    const auto source = fixture.emitter->GetUserData(IR::ScalarReg::S2);
    IR::Inst* current = PointerPhi(fixture, {source, UndefPointer(fixture)});

    // Every node has one concrete anchor and one predecessor, so this is still one unique source.
    // A malformed or adversarial shader must not make the matcher allocate an unbounded traversal
    // queue. The exact bound is an implementation detail; this exceeds the current conservative
    // provenance budget and must therefore be refused.
    for (size_t depth = 1; depth < 300; ++depth) {
        current = PointerPhi(fixture, {source, IR::Value{current}});
    }

    EXPECT_EQ(Detail::FindUniquePointerSource(*current), nullptr);
}

TEST(DynamicImageTableMatcher, UndefRefinementStillRequiresAnExactLowHighTablePair) {
    for (const bool pointer_form : {false, true}) {
        SCOPED_TRACE(pointer_form);
        DynamicImageTableFixture fixture;
        auto* const sample = pointer_form ? fixture.MakePointerSample() : fixture.MakeSample();
        auto* const image = sample->Arg(0).InstRecursive();
        auto* const pointer = image->Arg(0).InstRecursive();
        auto* const low = pointer_form ? pointer->Arg(0).InstRecursive() : pointer;
        auto* const high = pointer_form ? pointer->Arg(1).InstRecursive() : pointer;
        const u32 low_arg = pointer_form ? 1U : 0U;
        const u32 high_arg = pointer_form ? 0U : 1U;
        const auto low_source = low->Arg(low_arg);
        const auto high_source = high->Arg(high_arg);
        const auto undef = UndefPointer(fixture);
        low->SetArg(low_arg, IR::Value{PointerPhi(fixture, {low_source, undef})});
        high->SetArg(high_arg, IR::Value{PointerPhi(fixture, {high_source, undef})});

        const auto accepted = MatchDynamicImageTable(*sample);
        ASSERT_TRUE(accepted.has_value());
        EXPECT_EQ(accepted->pointer_lo, low_source.InstRecursive());
        EXPECT_EQ(accepted->pointer_hi, high_source.InstRecursive());

        const auto wrong_high = fixture.emitter->ReadConst(high_source.InstRecursive()->Arg(0),
                                                           fixture.emitter->Imm32(4U));
        high->SetArg(high_arg, IR::Value{PointerPhi(fixture, {wrong_high, undef})});
        EXPECT_FALSE(MatchDynamicImageTable(*sample).has_value());
    }
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

namespace Vulkan {
namespace {

struct GraphicsLayoutFixture {
    static constexpr std::array Types{
        vk::DescriptorType::eUniformBuffer, vk::DescriptorType::eStorageBuffer,
        vk::DescriptorType::eSampledImage, vk::DescriptorType::eStorageImage,
        vk::DescriptorType::eSampler};
    vk::PhysicalDeviceLimits limits{};
    std::array<vk::DescriptorPoolSize, Types.size()> pool{};
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    u32 max_sets{1024U};
    u32 colors{};
    u32 outputs{};
    bool push{};

    GraphicsLayoutFixture() {
        limits.maxBoundDescriptorSets = 1U;
        limits.maxPerStageDescriptorUniformBuffers = limits.maxDescriptorSetUniformBuffers = 4096U;
        limits.maxPerStageDescriptorStorageBuffers = limits.maxDescriptorSetStorageBuffers = 4096U;
        limits.maxPerStageDescriptorSampledImages = limits.maxDescriptorSetSampledImages = 4096U;
        limits.maxPerStageDescriptorStorageImages = limits.maxDescriptorSetStorageImages = 4096U;
        limits.maxPerStageDescriptorSamplers = limits.maxDescriptorSetSamplers = 4096U;
        limits.maxPerStageResources = limits.maxFragmentCombinedOutputResources = 4096U;
        for (size_t i = 0; i < Types.size(); ++i) {
            pool[i] = {.type = Types[i], .descriptorCount = 32U * 64U};
        }
    }

    void Add(vk::DescriptorType type, u32 count,
             vk::ShaderStageFlags stages = vk::ShaderStageFlagBits::eVertex) {
        bindings.push_back({.binding = static_cast<u32>(bindings.size()),
                            .descriptorType = type,
                            .descriptorCount = count,
                            .stageFlags = stages});
    }

    auto Validate() const {
        return ValidateGraphicsDescriptorLayout(bindings, limits, pool, max_sets, 32U, push, colors,
                                                outputs);
    }
};

TEST(GraphicsDescriptorLayout, CompleteStageArraysShareOneGlobalBudget) {
    GraphicsLayoutFixture fixture;
    fixture.Add(vk::DescriptorType::eSampledImage, 64U);
    fixture.Add(vk::DescriptorType::eSampledImage, 1U, vk::ShaderStageFlagBits::eFragment);
    fixture.limits.maxDescriptorSetSampledImages = 64U;
    const auto failure = fixture.Validate();
    ASSERT_TRUE(failure);
    EXPECT_EQ(failure->reason, DescriptorLayoutFailureReason::PipelineType);
    EXPECT_EQ(failure->required, 65U);
}

TEST(GraphicsDescriptorLayout, MultiStageVisibilityCountsOnceGloballyAndOncePerStage) {
    GraphicsLayoutFixture fixture;
    fixture.Add(vk::DescriptorType::eSampledImage, 64U,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
    fixture.limits.maxDescriptorSetSampledImages = 64U;
    fixture.limits.maxPerStageDescriptorSampledImages = 63U;
    ASSERT_TRUE(fixture.Validate());
    EXPECT_EQ(fixture.Validate()->reason, DescriptorLayoutFailureReason::PerStageType);
    fixture.limits.maxPerStageDescriptorSampledImages = 64U;
    EXPECT_FALSE(fixture.Validate());
}

TEST(GraphicsDescriptorLayout, StandaloneSamplersDoNotConsumePerStageResources) {
    GraphicsLayoutFixture fixture;
    fixture.Add(vk::DescriptorType::eSampledImage, 64U);
    fixture.Add(vk::DescriptorType::eSampler, 16U);
    fixture.limits.maxPerStageResources = 64U;
    EXPECT_FALSE(fixture.Validate());
    fixture.Add(vk::DescriptorType::eStorageBuffer, 1U);
    ASSERT_TRUE(fixture.Validate());
    EXPECT_EQ(fixture.Validate()->reason, DescriptorLayoutFailureReason::PerStageResources);
}

TEST(GraphicsDescriptorLayout, FragmentFramebufferColorsConsumeResourceCapacity) {
    GraphicsLayoutFixture fixture;
    fixture.Add(vk::DescriptorType::eSampledImage, 63U, vk::ShaderStageFlagBits::eFragment);
    fixture.colors = 2U;
    fixture.limits.maxPerStageResources = 64U;
    ASSERT_TRUE(fixture.Validate());
    EXPECT_EQ(fixture.Validate()->reason, DescriptorLayoutFailureReason::PerStageResources);
    fixture.colors = 1U;
    EXPECT_FALSE(fixture.Validate());
}

TEST(GraphicsDescriptorLayout, FragmentStorageAndOutputLocationsShareTheirOwnLimit) {
    GraphicsLayoutFixture fixture;
    fixture.Add(vk::DescriptorType::eStorageBuffer, 8U, vk::ShaderStageFlagBits::eFragment);
    fixture.Add(vk::DescriptorType::eStorageImage, 4U, vk::ShaderStageFlagBits::eFragment);
    fixture.outputs = 2U;
    fixture.limits.maxFragmentCombinedOutputResources = 13U;
    ASSERT_TRUE(fixture.Validate());
    EXPECT_EQ(fixture.Validate()->reason, DescriptorLayoutFailureReason::FragmentOutputs);
}

TEST(GraphicsDescriptorLayout, EveryActualDescriptorTypeMustFitTheWholeAllocationBatch) {
    for (const auto type : GraphicsLayoutFixture::Types) {
        SCOPED_TRACE(static_cast<u32>(type));
        GraphicsLayoutFixture fixture;
        fixture.Add(type, 65U);
        ASSERT_TRUE(fixture.Validate());
        EXPECT_EQ(fixture.Validate()->reason, DescriptorLayoutFailureReason::PoolType);
        fixture.bindings.front().descriptorCount = 64U;
        EXPECT_FALSE(fixture.Validate());
        fixture.pool.fill({});
        fixture.push = true;
        EXPECT_FALSE(fixture.Validate()); // Push descriptors do not allocate from this heap.
    }
}

TEST(GraphicsDescriptorLayout, FreshPoolMustFitBatchSetCountToo) {
    GraphicsLayoutFixture fixture;
    fixture.Add(vk::DescriptorType::eSampledImage, 1U);
    fixture.max_sets = 31U;
    ASSERT_TRUE(fixture.Validate());
    EXPECT_EQ(fixture.Validate()->reason, DescriptorLayoutFailureReason::PoolSets);
    fixture.max_sets = 32U;
    EXPECT_FALSE(fixture.Validate());
}

TEST(GraphicsDescriptorLayout, ImmutableSamplersStillCountAgainstSamplerLimits) {
    GraphicsLayoutFixture fixture;
    const std::array<vk::Sampler, 2> immutable{};
    fixture.Add(vk::DescriptorType::eSampler, 2U);
    fixture.bindings.front().pImmutableSamplers = immutable.data();
    fixture.limits.maxPerStageDescriptorSamplers = 1U;
    ASSERT_TRUE(fixture.Validate());
    EXPECT_EQ(fixture.Validate()->reason, DescriptorLayoutFailureReason::PerStageType);
    fixture.limits.maxPerStageDescriptorSamplers = 2U;
    fixture.limits.maxDescriptorSetSamplers = 1U;
    EXPECT_EQ(fixture.Validate()->reason, DescriptorLayoutFailureReason::PipelineType);
}

TEST(GraphicsDescriptorLayout, ZeroCountsUnsupportedStagesAndDuplicateBindingsRefuse) {
    GraphicsLayoutFixture fixture;
    fixture.Add(vk::DescriptorType::eStorageBuffer, 0U);
    ASSERT_TRUE(fixture.Validate());
    EXPECT_EQ(fixture.Validate()->reason, DescriptorLayoutFailureReason::MalformedBinding);
    fixture.bindings.front().descriptorCount = 1U;
    fixture.bindings.front().stageFlags = {};
    EXPECT_TRUE(fixture.Validate());
    fixture.bindings.front().stageFlags = vk::ShaderStageFlagBits::eCompute;
    EXPECT_TRUE(fixture.Validate());
    fixture.bindings.front().stageFlags = vk::ShaderStageFlagBits::eVertex;
    fixture.bindings.push_back(fixture.bindings.front());
    EXPECT_TRUE(fixture.Validate());
}

TEST(GraphicsDescriptorLayout, AggregateCountsNeverWrapAtU32) {
    GraphicsLayoutFixture fixture;
    fixture.Add(vk::DescriptorType::eSampledImage, std::numeric_limits<u32>::max());
    fixture.Add(vk::DescriptorType::eSampledImage, 1U);
    fixture.limits.maxPerStageDescriptorSampledImages = std::numeric_limits<u32>::max();
    fixture.limits.maxDescriptorSetSampledImages = std::numeric_limits<u32>::max();
    const auto failure = fixture.Validate();
    ASSERT_TRUE(failure);
    EXPECT_EQ(failure->required, u64{std::numeric_limits<u32>::max()} + 1U);
}

TEST(GraphicsDescriptorLayout, UnsupportedLayoutTypesRequireExplicitFutureAccounting) {
    GraphicsLayoutFixture fixture;
    fixture.Add(vk::DescriptorType::eCombinedImageSampler, 1U);
    ASSERT_TRUE(fixture.Validate());
    EXPECT_EQ(fixture.Validate()->reason, DescriptorLayoutFailureReason::UnsupportedType);
}

TEST(GraphicsDescriptorLayout, EmptyLayoutNeedsNoDescriptorPoolResourcesWhenPushed) {
    GraphicsLayoutFixture fixture;
    fixture.push = true;
    fixture.pool.fill({});
    EXPECT_FALSE(fixture.Validate());
}

} // namespace
} // namespace Vulkan
