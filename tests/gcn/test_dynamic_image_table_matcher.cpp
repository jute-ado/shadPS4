// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#ifdef _WIN32
#include <process.h>
#endif

#include <gtest/gtest.h>

#include "common/serdes.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/dynamic_image_table_matcher.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/specialization.h"
#include "spirv/unified1/spirv.hpp"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

namespace Shader::Optimization {
namespace {

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

    IR::Value MakeTablePointer() {
        auto& ir = *emitter;
        const auto root = ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0),
                                                ir.GetUserData(IR::ScalarReg::S1));
        return ir.CompositeConstruct(ir.ReadConst(root, ir.Imm32(2U)),
                                     ir.ReadConst(root, ir.Imm32(3U)));
    }

    IR::Inst* MakeAccess(bool explicit_lod = true, u32 byte_offset = 14480, u32 byte_stride = 32,
                         bool is_r128 = true) {
        auto& ir = *emitter;
        const auto scaled_index = ir.IMul(ir.LaneId(), ir.Imm32(byte_stride));
        const auto byte_address = ir.IAdd(scaled_index, ir.Imm32(byte_offset));
        const auto dword_offset = ir.ShiftRightLogical(byte_address, ir.Imm32(2U));
        const auto image = ir.ReadConst(MakeTablePointer(), dword_offset);
        IR::TextureInstInfo texture_info{};
        texture_info.has_lod.Assign(explicit_lod);
        texture_info.is_r128.Assign(is_r128);
        const auto access = ir.ImageSampleRaw(image, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U),
                                              ir.Imm32(0U), ir.Imm32(0U), texture_info);
        ir.Reference(access);
        return access.InstRecursive();
    }

    IR::Inst* MakePatchableAccessWithIndex(IR::U32 index, u32 inline_sampler_raw0_lo = 0U,
                                           bool is_array = false, bool is_r128 = true) {
        auto& ir = *emitter;
        const auto scaled_index = ir.IMul(index, ir.Imm32(32U));
        const auto byte_address = ir.IAdd(scaled_index, ir.Imm32(14480U));
        const auto dword_offset = ir.ShiftRightLogical(byte_address, ir.Imm32(2U));
        const auto image = ir.ReadConst(MakeTablePointer(), dword_offset);
        const auto sampler =
            ir.CompositeConstruct(ir.Imm32(inline_sampler_raw0_lo), ir.Imm32(0U), ir.Imm32(0U),
                                  ir.Imm32(0U));
        const auto coords_lod = ir.CompositeConstruct(ir.Imm32(0.25f), ir.Imm32(0.5f),
                                                      ir.Imm32(0.0f), ir.Imm32(0.0f));
        const auto unused = ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f),
                                                  ir.Imm32(0.0f), ir.Imm32(0.0f));
        IR::TextureInstInfo texture_info{};
        texture_info.has_lod.Assign(true);
        texture_info.is_r128.Assign(is_r128);
        texture_info.is_array.Assign(is_array);
        const auto access = ir.ImageSampleRaw(image, sampler, coords_lod, unused, unused,
                                              ir.Imm32(0.0f), texture_info);
        ir.Reference(access);
        return access.InstRecursive();
    }

    IR::Inst* MakePatchableAccess(u32 inline_sampler_raw0_lo = 0U, bool is_array = false,
                                  bool is_r128 = true) {
        return MakePatchableAccessWithIndex(emitter->LaneId(), inline_sampler_raw0_lo, is_array,
                                            is_r128);
    }
};

Profile DynamicImageProfile(u32 limit = 256U) {
    Profile profile{};
    profile.max_sampled_images_per_stage = limit;
    profile.max_sampled_images_per_set = limit;
    profile.max_resources_per_stage = limit;
    profile.max_sampled_images_per_descriptor_heap_set = limit;
    profile.supports_sampled_image_array_nonuniform_indexing = true;
    return profile;
}

bool PS4_SYSV_ABI ReadDynamicImageTableRange(void*, VAddr address, void* output, u64 size) {
    std::memcpy(output, reinterpret_cast<const void*>(address), size);
    return true;
}

size_t CountSpirvOpcode(const std::vector<u32>& spirv, spv::Op opcode) {
    size_t count{};
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 instruction = spirv[offset];
        const u32 word_count = instruction >> 16;
        EXPECT_GT(word_count, 0U);
        if (static_cast<spv::Op>(instruction & 0xffffU) == opcode) {
            ++count;
        }
        offset += word_count;
    }
    return count;
}

bool HasSpirvCapability(const std::vector<u32>& spirv, spv::Capability capability) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 instruction = spirv[offset];
        const u32 word_count = instruction >> 16;
        if (static_cast<spv::Op>(instruction & 0xffffU) == spv::OpCapability &&
            word_count >= 2U && spirv[offset + 1U] == static_cast<u32>(capability)) {
            return true;
        }
        offset += word_count;
    }
    return false;
}

bool HasSpirvArrayLength(const std::vector<u32>& spirv, u32 expected_length) {
    std::vector<u32> matching_constant_ids;
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 instruction = spirv[offset];
        const u32 word_count = instruction >> 16;
        if (static_cast<spv::Op>(instruction & 0xffffU) == spv::OpConstant &&
            word_count >= 4U && spirv[offset + 3U] == expected_length) {
            matching_constant_ids.push_back(spirv[offset + 2U]);
        }
        offset += word_count;
    }
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 instruction = spirv[offset];
        const u32 word_count = instruction >> 16;
        if (static_cast<spv::Op>(instruction & 0xffffU) == spv::OpTypeArray &&
            word_count >= 4U &&
            std::ranges::find(matching_constant_ids, spirv[offset + 3U]) !=
                matching_constant_ids.end()) {
            return true;
        }
        offset += word_count;
    }
    return false;
}

#ifdef SHADPS4_SPIRV_VAL_PATH
int RunSpirvValidator(const std::filesystem::path& module) {
#ifdef _WIN32
    const std::string module_path = module.string();
    return static_cast<int>(_spawnl(_P_WAIT, SHADPS4_SPIRV_VAL_PATH,
                                    SHADPS4_SPIRV_VAL_PATH, "--target-env", "vulkan1.3",
                                    module_path.c_str(), nullptr));
#else
    const std::string command = std::string{"\""} + SHADPS4_SPIRV_VAL_PATH +
                                "\" --target-env vulkan1.3 \"" + module.string() + "\"";
    return std::system(command.c_str());
#endif
}
#endif

void InstallDynamicImageTable(DynamicImageTableFixture& fixture,
                              const DynamicImageTableMatch& match, u32 destination,
                              AmdGpu::Image model = AmdGpu::Image::Null(false)) {
    const u32 descriptor_dw = match.is_r128 ? 4U : sizeof(AmdGpu::Image) / sizeof(u32);
    const u32 snapshot_span_dw = (NUM_IMAGES - 1U) * 8U + descriptor_dw;
    const u32 required_size = destination + snapshot_span_dw;
    fixture.info.flattened_ud_buf.resize(
        std::max<size_t>(fixture.info.flattened_ud_buf.size(), required_size));
    fixture.info.srt_info.flattened_bufsize_dw =
        std::max(fixture.info.srt_info.flattened_bufsize_dw, required_size);
    model.base_address = model.base_address != 0 ? model.base_address : 0x100U;
    model.width = model.width != 0 ? model.width : 63U;
    model.height = model.height != 0 ? model.height : 63U;
    std::memcpy(fixture.info.flattened_ud_buf.data() + destination, &model,
                descriptor_dw * sizeof(u32));
    match.pointer_lo->SetFlags<u32>(18U);
    match.pointer_hi->SetFlags<u32>(19U);
    match.image_handle->SetFlags<u32>(destination);
    const PersistentSrtInfo::SrtRangeReservation reservation{
        .pointer_lo_flat_index = 18U,
        .source_offset_dw = 3620U,
        .destination_flat_offset_dw = destination,
        .span_dw = snapshot_span_dw,
    };
    if (std::ranges::find(fixture.info.srt_info.range_reservations, reservation) ==
        fixture.info.srt_info.range_reservations.end()) {
        fixture.info.srt_info.range_reservations.push_back(reservation);
    }
}

TEST(DynamicImageTableMatcher, RecognizesTheUnboundedAffineExplicitLodShape) {
    DynamicImageTableFixture fixture;
    IR::Inst* const access = fixture.MakeAccess();

    const auto match = MatchDynamicImageTable(*access);
    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(match->byte_offset, 14480U);
    EXPECT_EQ(match->byte_stride, 32U);
    EXPECT_EQ(match->access, DynamicImageAccess::SampledExplicitLod);
    EXPECT_TRUE(match->is_r128);
    EXPECT_FALSE(match->index_upper_bound.has_value());
    ASSERT_NE(match->image_handle, nullptr);
    ASSERT_NE(match->pointer_lo, nullptr);
    ASSERT_NE(match->pointer_hi, nullptr);

    const auto plan = PlanDynamicImageTable(match->Pattern(
        18U, NUM_IMAGES, true /* nonuniform indexing */, true /* shader OOB guard */));
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->base_pointer_location, 18U);
    EXPECT_EQ(plan->offset_dw, 3620U);
    EXPECT_EQ(plan->stride_dw, 8U);
    EXPECT_EQ(plan->descriptor_count, NUM_IMAGES);
    EXPECT_EQ(plan->snapshot_span_dw, (NUM_IMAGES - 1U) * 8U + 4U);
    EXPECT_TRUE(plan->requires_oob_guard);
}

TEST(DynamicImageTableMatcher, RecognizesPointerWordsThroughLoopCarriedMaskMaterialization) {
    DynamicImageTableFixture fixture;
    auto& ir = *fixture.emitter;

    const auto root = ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0),
                                            ir.GetUserData(IR::ScalarReg::S1));
    const auto pointer_lo = ir.ReadConst(root, ir.Imm32(2U));
    const auto pointer_hi = ir.ReadConst(root, ir.Imm32(3U));

    IR::Inst* const lo_phi =
        &*fixture.block->PrependNewInst(fixture.block->begin(), IR::Opcode::Phi);
    lo_phi->SetFlags(IR::Type::U32);
    lo_phi->AddPhiOperand(fixture.block, pointer_lo);
    const auto materialized_lo =
        ir.MaterializeScalarMaskWord(ir.Imm1(false), ir.Imm1(false), IR::U32{lo_phi}, 0U);
    lo_phi->AddPhiOperand(fixture.block, materialized_lo);

    IR::Inst* const hi_phi =
        &*fixture.block->PrependNewInst(fixture.block->begin(), IR::Opcode::Phi);
    hi_phi->SetFlags(IR::Type::U32);
    hi_phi->AddPhiOperand(fixture.block, pointer_hi);
    const auto materialized_hi =
        ir.MaterializeScalarMaskWord(ir.Imm1(false), ir.Imm1(false), IR::U32{hi_phi}, 1U);
    hi_phi->AddPhiOperand(fixture.block, materialized_hi);

    const auto table_pointer = ir.CompositeConstruct(materialized_lo, materialized_hi);
    const auto scaled_index = ir.IMul(ir.LaneId(), ir.Imm32(32U));
    const auto byte_address = ir.IAdd(scaled_index, ir.Imm32(14480U));
    const auto dword_offset = ir.ShiftRightLogical(byte_address, ir.Imm32(2U));
    const auto image = ir.ReadConst(table_pointer, dword_offset);
    IR::TextureInstInfo texture_info{};
    texture_info.has_lod.Assign(true);
    texture_info.is_r128.Assign(true);
    const auto access = ir.ImageSampleRaw(image, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U),
                                          ir.Imm32(0U), ir.Imm32(0U), texture_info);
    ir.Reference(access);

    const auto match = MatchDynamicImageTable(*access.InstRecursive());
    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(match->pointer_lo, pointer_lo.InstRecursive());
    EXPECT_EQ(match->pointer_hi, pointer_hi.InstRecursive());
    EXPECT_EQ(match->byte_offset, 14480U);
    EXPECT_EQ(match->byte_stride, 32U);
}

TEST(DynamicImageTableMatcher, DeclinesPointerWordsThatCanMaterializeAsMasks) {
    DynamicImageTableFixture fixture;
    auto& ir = *fixture.emitter;

    const auto root = ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0),
                                            ir.GetUserData(IR::ScalarReg::S1));
    const auto pointer_lo = ir.ReadConst(root, ir.Imm32(2U));
    const auto pointer_hi = ir.ReadConst(root, ir.Imm32(3U));
    const auto mask_lo =
        ir.MaterializeScalarMaskWord(ir.Imm1(true), ir.Imm1(true), pointer_lo, 0U);
    const auto mask_hi =
        ir.MaterializeScalarMaskWord(ir.Imm1(true), ir.Imm1(true), pointer_hi, 1U);
    const auto table_pointer = ir.CompositeConstruct(mask_lo, mask_hi);
    const auto scaled_index = ir.IMul(ir.LaneId(), ir.Imm32(32U));
    const auto byte_address = ir.IAdd(scaled_index, ir.Imm32(14480U));
    const auto dword_offset = ir.ShiftRightLogical(byte_address, ir.Imm32(2U));
    const auto image = ir.ReadConst(table_pointer, dword_offset);
    IR::TextureInstInfo texture_info{};
    texture_info.has_lod.Assign(true);
    texture_info.is_r128.Assign(true);
    const auto access = ir.ImageSampleRaw(image, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U),
                                          ir.Imm32(0U), ir.Imm32(0U), texture_info);
    ir.Reference(access);

    EXPECT_FALSE(MatchDynamicImageTable(*access.InstRecursive()).has_value());
}

TEST(DynamicImageTableMatcher, DeclinesImplicitLodAndMalformedAffineOffsets) {
    DynamicImageTableFixture implicit_fixture;
    const auto implicit = MatchDynamicImageTable(*implicit_fixture.MakeAccess(false));
    ASSERT_TRUE(implicit.has_value());
    EXPECT_EQ(implicit->access, DynamicImageAccess::SampledImplicitLod);
    EXPECT_FALSE(PlanDynamicImageTable(implicit->Pattern(18U, NUM_IMAGES, true, true)).has_value());

    DynamicImageTableFixture unaligned_fixture;
    const auto unaligned = MatchDynamicImageTable(*unaligned_fixture.MakeAccess(true, 14481));
    ASSERT_TRUE(unaligned.has_value());
    EXPECT_FALSE(
        PlanDynamicImageTable(unaligned->Pattern(18U, NUM_IMAGES, true, true)).has_value());

    DynamicImageTableFixture bad_stride_fixture;
    const auto bad_stride = MatchDynamicImageTable(*bad_stride_fixture.MakeAccess(true, 14480, 48));
    ASSERT_TRUE(bad_stride.has_value());
    EXPECT_FALSE(
        PlanDynamicImageTable(bad_stride->Pattern(18U, NUM_IMAGES, true, true)).has_value());
}

TEST(DynamicImageTableMatcher, DeclinesRawShapesThatDoNotLowerToPlainExplicitLodSampling) {
    const auto is_rejected = [](auto mutate_info) {
        DynamicImageTableFixture fixture;
        IR::Inst* const access = fixture.MakeAccess();
        auto info = access->Flags<IR::TextureInstInfo>();
        mutate_info(info);
        access->SetFlags(info);
        return !MatchDynamicImageTable(*access).has_value();
    };

    EXPECT_TRUE(is_rejected([](auto& info) { info.is_depth.Assign(true); }));
    EXPECT_TRUE(is_rejected([](auto& info) { info.is_gather.Assign(true); }));
    EXPECT_TRUE(is_rejected([](auto& info) { info.has_derivatives.Assign(true); }));
    EXPECT_TRUE(is_rejected([](auto& info) { info.has_lod_clamp.Assign(true); }));
    EXPECT_TRUE(is_rejected([](auto& info) { info.is_unnormalized.Assign(true); }));
    EXPECT_FALSE(is_rejected([](auto& info) { info.has_offset.Assign(true); }));
    EXPECT_FALSE(is_rejected([](auto& info) { info.is_array.Assign(true); }));
    EXPECT_FALSE(is_rejected([](auto& info) {
        info.has_lod.Assign(false);
        info.force_level0.Assign(true);
    }));
}

TEST(DynamicImageTableMatcher, DerivesAProvenBoundFromAnIndexMask) {
    DynamicImageTableFixture fixture;
    auto& ir = *fixture.emitter;
    const auto bounded_index = ir.BitwiseAnd(ir.LaneId(), ir.Imm32(5U));
    const auto scaled_index = ir.IMul(bounded_index, ir.Imm32(32U));
    const auto byte_address = ir.IAdd(scaled_index, ir.Imm32(14480U));
    const auto dword_offset = ir.ShiftRightLogical(byte_address, ir.Imm32(2U));
    const auto image = ir.ReadConst(fixture.MakeTablePointer(), dword_offset);
    IR::TextureInstInfo texture_info{};
    texture_info.has_lod.Assign(true);
    texture_info.is_r128.Assign(true);
    const auto access = ir.ImageSampleRaw(image, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U),
                                          ir.Imm32(0U), ir.Imm32(0U), texture_info);
    ir.Reference(access);

    const auto match = MatchDynamicImageTable(*access.InstRecursive());
    ASSERT_TRUE(match.has_value());
    ASSERT_TRUE(match->index_upper_bound.has_value());
    EXPECT_EQ(*match->index_upper_bound, 5U);
    const auto plan = PlanDynamicImageTable(match->Pattern(18U, NUM_IMAGES, true, true));
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->descriptor_count, 6U);
    EXPECT_FALSE(plan->requires_oob_guard);
}

TEST(DynamicImageTableBudget, AppliesEveryLimitWithSaturatingSubtraction) {
    EXPECT_EQ(DynamicImageDescriptorWindow(64U, 0U, 0U, 256U, 256U, 256U, 256U),
              64U);
    EXPECT_EQ(DynamicImageDescriptorWindow(64U, 8U, 9U, 40U, 256U, 256U, 256U),
              32U);
    EXPECT_EQ(DynamicImageDescriptorWindow(64U, 8U, 9U, 256U, 28U, 256U, 256U),
              20U);
    EXPECT_EQ(DynamicImageDescriptorWindow(64U, 8U, 9U, 256U, 256U, 24U, 256U),
              15U);
    EXPECT_EQ(DynamicImageDescriptorWindow(64U, 8U, 9U, 256U, 256U, 256U, 18U),
              10U);
    EXPECT_EQ(DynamicImageDescriptorWindow(64U, 257U, 9U, 256U, 256U, 256U, 256U),
              0U);
    EXPECT_EQ(DynamicImageDescriptorWindow(64U, 8U, 300U, 256U, 256U, 256U, 256U),
              0U);
}

TEST(DynamicImageTablePolicy, RejectsGuestByteAddressWraparound) {
    const DynamicImageTablePattern wrapping{
        .base_pointer_location = 18U,
        .byte_offset = 0xFFFFFFF0U,
        .byte_stride = 32U,
        .is_r128 = true,
        .access = DynamicImageAccess::SampledExplicitLod,
        .index_upper_bound = 1U,
        .supported_descriptor_window = 2U,
        .supports_nonuniform_indexing = true,
        .supports_oob_guard = true,
    };
    EXPECT_FALSE(PlanDynamicImageTable(wrapping).has_value());

    auto boundary = wrapping;
    boundary.byte_offset = 0xFFFFFFC0U;
    ASSERT_TRUE(PlanDynamicImageTable(boundary).has_value());
    EXPECT_EQ(PlanDynamicImageTable(boundary)->descriptor_count, 2U);
}

TEST(DynamicImageTableResourceTracking, PreservesTheRuntimeSelectorIntoTheGuardedSampleOpcode) {
    DynamicImageTableFixture fixture;
    IR::Inst* const access = fixture.MakePatchableAccess();
    const auto match = MatchDynamicImageTable(*access);
    ASSERT_TRUE(match.has_value());

    constexpr u32 TableDestination = 20U;
    InstallDynamicImageTable(fixture, *match, TableDestination);
    ResourceTrackingPass(fixture.program, DynamicImageProfile());

    ASSERT_EQ(fixture.info.images.size(), 1U);
    const auto& resource = fixture.info.images.front();
    EXPECT_EQ(resource.sharp_idx, TableDestination);
    EXPECT_EQ(resource.dynamic_descriptor_count, NUM_IMAGES);
    EXPECT_EQ(resource.dynamic_descriptor_stride_dw, 8U);
    EXPECT_TRUE(resource.requires_dynamic_oob_guard);
    EXPECT_FALSE(resource.is_written);
    EXPECT_FALSE(resource.is_atomic);

    const auto dynamic_sample = std::ranges::find_if(
        fixture.block->Instructions(), [](const IR::Inst& inst) {
            return inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLodDynamic;
        });
    ASSERT_NE(dynamic_sample, fixture.block->Instructions().end());
    ASSERT_FALSE(dynamic_sample->Arg(1).IsImmediate());
    EXPECT_EQ(dynamic_sample->Arg(1).InstRecursive()->GetOpcode(), IR::Opcode::LaneId);
}

TEST(DynamicImageTableResourceTracking,
     Fb974Shadow256BitShapeReservesAndTracksEveryDescriptorDword) {
    DynamicImageTableFixture fixture;
    // Accepted fb974 at PC 0xe60: affine byte offset 14480, byte stride 32, R128=0.
    IR::Inst* const access = fixture.MakePatchableAccess(0U, false, false);
    const auto match = MatchDynamicImageTable(*access);
    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(match->byte_offset, 14480U);
    EXPECT_EQ(match->byte_stride, 32U);
    EXPECT_EQ(match->access, DynamicImageAccess::SampledExplicitLod);
    EXPECT_FALSE(match->is_r128);

    const auto plan = PlanDynamicImageTable(match->Pattern(
        18U, NUM_IMAGES, true /* nonuniform indexing */, true /* shader OOB guard */));
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->stride_dw, 8U);
    EXPECT_EQ(plan->snapshot_span_dw, 512U);

    constexpr u32 TableDestination = 20U;
    InstallDynamicImageTable(fixture, *match, TableDestination);
    ASSERT_EQ(fixture.info.srt_info.range_reservations.size(), 1U);
    EXPECT_EQ(fixture.info.srt_info.range_reservations.front().span_dw, 512U);
    EXPECT_GE(fixture.info.flattened_ud_buf.size(), TableDestination + 512U);

    ResourceTrackingPass(fixture.program, DynamicImageProfile());

    ASSERT_EQ(fixture.info.images.size(), 1U);
    const auto& resource = fixture.info.images.front();
    EXPECT_FALSE(resource.is_r128);
    EXPECT_EQ(resource.dynamic_descriptor_count, NUM_IMAGES);
    EXPECT_EQ(resource.dynamic_descriptor_stride_dw, 8U);
}

TEST(DynamicImageTableResourceTracking, RejectsAStampedReadWithoutTheReservedTableSnapshot) {
    DynamicImageTableFixture fixture;
    IR::Inst* const access = fixture.MakePatchableAccess();
    const auto match = MatchDynamicImageTable(*access);
    ASSERT_TRUE(match.has_value());

    // A normal flattened ReadConst also carries a destination flag. That flag alone must never be
    // treated as proof that FlattenExtendedUserdataPass reserved an entire descriptor table.
    constexpr u32 OrdinaryReadDestination = 20U;
    fixture.info.flattened_ud_buf.resize(OrdinaryReadDestination + 1U);
    match->image_handle->SetFlags<u32>(OrdinaryReadDestination);

    ResourceTrackingPass(fixture.program, DynamicImageProfile());

    EXPECT_TRUE(fixture.info.images.empty());
    EXPECT_EQ(std::ranges::count_if(fixture.block->Instructions(), [](const IR::Inst& inst) {
                  return inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLodDynamic;
              }),
              0U);
}

TEST(DynamicImageTableResourceTracking, ReusesOneDescriptorArrayForDuplicateTableAccesses) {
    DynamicImageTableFixture fixture;
    IR::Inst* const first_access = fixture.MakePatchableAccess();
    IR::Inst* const second_access = fixture.MakePatchableAccess();
    const auto first_match = MatchDynamicImageTable(*first_access);
    const auto second_match = MatchDynamicImageTable(*second_access);
    ASSERT_TRUE(first_match.has_value());
    ASSERT_TRUE(second_match.has_value());
    InstallDynamicImageTable(fixture, *first_match, 20U);
    InstallDynamicImageTable(fixture, *second_match, 20U);

    ResourceTrackingPass(fixture.program, DynamicImageProfile());

    ASSERT_EQ(fixture.info.images.size(), 1U);
    EXPECT_EQ(fixture.info.images.front().dynamic_descriptor_count, NUM_IMAGES);
    EXPECT_EQ(std::ranges::count_if(fixture.block->Instructions(), [](const IR::Inst& inst) {
                  return inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLodDynamic;
              }),
              2U);
}

TEST(DynamicImageTableResourceTracking,
     FullFlattenDeduplicatesDistinctEquivalentAffineTableHandles) {
    DynamicImageTableFixture fixture;
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    std::array<u32, 8> root{};
    std::array<u32, 4200> table_storage{};
    fixture.info.user_data = user_data;

    const VAddr root_address = reinterpret_cast<VAddr>(root.data());
    const VAddr table_address = reinterpret_cast<VAddr>(table_storage.data());
    user_data[0] = static_cast<u32>(root_address);
    user_data[1] = static_cast<u32>(root_address >> 32);
    root[2] = static_cast<u32>(table_address);
    root[3] = static_cast<u32>(table_address >> 32);

    constexpr u32 TableOffsetDw = 14480U / sizeof(u32);
    AmdGpu::Image model = AmdGpu::Image::Null(false);
    model.base_address = 0x100U;
    model.width = 63U;
    model.height = 63U;
    std::memcpy(table_storage.data() + TableOffsetDw, &model, sizeof(u128));

    IR::Inst* const first_access = fixture.MakePatchableAccess();
    IR::Inst* const second_access = fixture.MakePatchableAccess();
    const auto first_match = MatchDynamicImageTable(*first_access);
    const auto second_match = MatchDynamicImageTable(*second_access);
    ASSERT_TRUE(first_match.has_value());
    ASSERT_TRUE(second_match.has_value());
    ASSERT_NE(first_match->image_handle, second_match->image_handle);

    const SrtMemoryReader reader{ReadDynamicImageTableRange, nullptr};
    FlattenExtendedUserdataPass(fixture.program, reader);

    EXPECT_EQ(fixture.info.srt_info.range_reservations.size(), 1U);
    EXPECT_EQ(first_match->image_handle->Flags<u32>(), second_match->image_handle->Flags<u32>());

    ResourceTrackingPass(fixture.program, DynamicImageProfile(NUM_IMAGES + 1U));

    ASSERT_EQ(fixture.info.images.size(), 1U);
    EXPECT_EQ(fixture.info.images.front().dynamic_descriptor_count, NUM_IMAGES);
    EXPECT_EQ(fixture.info.samplers.size(), 1U);
    EXPECT_EQ(std::ranges::count_if(fixture.block->Instructions(), [](const IR::Inst& inst) {
                  return inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLodDynamic;
              }),
              2U);
}

TEST(DynamicImageTableResourceTracking,
     RejectsASecondSamplerWhenAReusedTableAlreadyFillsTheResourceBudget) {
    DynamicImageTableFixture fixture;
    IR::Inst* const first_access = fixture.MakePatchableAccess(0U);
    IR::Inst* const second_access = fixture.MakePatchableAccess(1U);
    const auto first_match = MatchDynamicImageTable(*first_access);
    const auto second_match = MatchDynamicImageTable(*second_access);
    ASSERT_TRUE(first_match.has_value());
    ASSERT_TRUE(second_match.has_value());
    InstallDynamicImageTable(fixture, *first_match, 20U);
    InstallDynamicImageTable(fixture, *second_match, 20U);

    auto profile = DynamicImageProfile();
    profile.max_resources_per_stage = NUM_IMAGES + 1U;
    ResourceTrackingPass(fixture.program, profile);

    ASSERT_EQ(fixture.info.images.size(), 1U);
    EXPECT_EQ(fixture.info.images.front().dynamic_descriptor_count, NUM_IMAGES);
    EXPECT_EQ(fixture.info.samplers.size(), 1U);
    EXPECT_EQ(std::ranges::count_if(fixture.block->Instructions(), [](const IR::Inst& inst) {
                  return inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLodDynamic;
              }),
              1U);
}

TEST(DynamicImageTableResourceTracking, GrowsAReusedTableForALaterWiderSelector) {
    DynamicImageTableFixture fixture;
    auto& ir = *fixture.emitter;
    const auto bounded_index = ir.BitwiseAnd(ir.LaneId(), ir.Imm32(0U));
    IR::Inst* const bounded_access = fixture.MakePatchableAccessWithIndex(bounded_index);
    IR::Inst* const unbounded_access = fixture.MakePatchableAccess();
    const auto bounded_match = MatchDynamicImageTable(*bounded_access);
    const auto unbounded_match = MatchDynamicImageTable(*unbounded_access);
    ASSERT_TRUE(bounded_match.has_value());
    ASSERT_TRUE(unbounded_match.has_value());
    ASSERT_EQ(bounded_match->index_upper_bound, 0U);

    constexpr u32 Destination = 20U;
    InstallDynamicImageTable(fixture, *unbounded_match, Destination);
    bounded_match->pointer_lo->SetFlags<u32>(18U);
    bounded_match->pointer_hi->SetFlags<u32>(19U);
    bounded_match->image_handle->SetFlags<u32>(Destination);
    fixture.info.srt_info.range_reservations.push_back({
        .pointer_lo_flat_index = 18U,
        .source_offset_dw = 3620U,
        .destination_flat_offset_dw = Destination,
        .span_dw = 4U,
    });

    ResourceTrackingPass(fixture.program, DynamicImageProfile());

    ASSERT_EQ(fixture.info.images.size(), 1U);
    EXPECT_EQ(fixture.info.images.front().dynamic_descriptor_count, NUM_IMAGES);
    EXPECT_TRUE(fixture.info.images.front().requires_dynamic_oob_guard);
    EXPECT_EQ(std::ranges::count_if(fixture.block->Instructions(), [](const IR::Inst& inst) {
                  return inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLodDynamic;
              }),
              2U);
}

TEST(DynamicImageTableResourceTracking, DistinctTablesConsumeOnlyTheRemainingDescriptorBudget) {
    DynamicImageTableFixture fixture;
    IR::Inst* const first_access = fixture.MakePatchableAccess();
    IR::Inst* const second_access = fixture.MakePatchableAccess();
    const auto first_match = MatchDynamicImageTable(*first_access);
    const auto second_match = MatchDynamicImageTable(*second_access);
    ASSERT_TRUE(first_match.has_value());
    ASSERT_TRUE(second_match.has_value());
    InstallDynamicImageTable(fixture, *first_match, 20U);
    InstallDynamicImageTable(fixture, *second_match, 20U + 508U);

    auto profile = DynamicImageProfile(80U);
    profile.max_resources_per_stage = 81U; // 80 sampled images plus the shared sampler.
    ResourceTrackingPass(fixture.program, profile);

    ASSERT_EQ(fixture.info.images.size(), 2U);
    EXPECT_EQ(fixture.info.images[0].dynamic_descriptor_count, 64U);
    EXPECT_EQ(fixture.info.images[1].dynamic_descriptor_count, 16U);
    EXPECT_LE(fixture.info.images[0].dynamic_descriptor_count +
                  fixture.info.images[1].dynamic_descriptor_count,
              80U);
}

TEST(DynamicImageTableResourceTracking, ReservesPerStageResourceSpaceForTheSampler) {
    DynamicImageTableFixture fixture;
    IR::Inst* const access = fixture.MakePatchableAccess();
    const auto match = MatchDynamicImageTable(*access);
    ASSERT_TRUE(match.has_value());
    InstallDynamicImageTable(fixture, *match, 20U);

    auto profile = DynamicImageProfile();
    profile.max_resources_per_stage = 64U;
    ResourceTrackingPass(fixture.program, profile);

    ASSERT_EQ(fixture.info.images.size(), 1U);
    EXPECT_EQ(fixture.info.images.front().dynamic_descriptor_count, 63U);
    EXPECT_EQ(fixture.info.samplers.size(), 1U);
}

TEST(DynamicImageTableResourceTracking, RejectsExhaustedAndNonComputeLayouts) {
    {
        DynamicImageTableFixture fixture;
        IR::Inst* const access = fixture.MakePatchableAccess();
        const auto match = MatchDynamicImageTable(*access);
        ASSERT_TRUE(match.has_value());
        InstallDynamicImageTable(fixture, *match, 20U);
        ResourceTrackingPass(fixture.program, DynamicImageProfile(0U));
        EXPECT_TRUE(fixture.info.images.empty());
    }
    {
        DynamicImageTableFixture fixture;
        fixture.info.stage = Stage::Fragment;
        fixture.info.l_stage = LogicalStage::Fragment;
        IR::Inst* const access = fixture.MakePatchableAccess();
        const auto match = MatchDynamicImageTable(*access);
        ASSERT_TRUE(match.has_value());
        InstallDynamicImageTable(fixture, *match, 20U);
        ResourceTrackingPass(fixture.program, DynamicImageProfile());
        EXPECT_TRUE(fixture.info.images.empty());
    }
}

TEST(DynamicImageTableResourceTracking,
     RetainsTheLatentLayoutWhileRejectingUnsupportedRuntimeDescriptors) {
    {
        DynamicImageTableFixture fixture;
        IR::Inst* const access = fixture.MakePatchableAccess();
        const auto match = MatchDynamicImageTable(*access);
        ASSERT_TRUE(match.has_value());
        auto msaa = AmdGpu::Image::Null(false);
        msaa.type = static_cast<u64>(AmdGpu::ImageType::Color2DMsaa);
        InstallDynamicImageTable(fixture, *match, 20U, msaa);
        ResourceTrackingPass(fixture.program, DynamicImageProfile());
        ASSERT_EQ(fixture.info.images.size(), 1U);
        EXPECT_EQ(fixture.info.images.front().dynamic_descriptor_count, NUM_IMAGES);
        EXPECT_EQ(fixture.info.samplers.size(), 1U);
        EXPECT_EQ(std::ranges::count_if(fixture.block->Instructions(), [](const IR::Inst& inst) {
                      return inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLodDynamic;
                  }),
                  0U);
    }
    {
        DynamicImageTableFixture fixture;
        IR::Inst* const access = fixture.MakePatchableAccess(1U << 15);
        const auto match = MatchDynamicImageTable(*access);
        ASSERT_TRUE(match.has_value());
        InstallDynamicImageTable(fixture, *match, 20U);
        ResourceTrackingPass(fixture.program, DynamicImageProfile());
        ASSERT_EQ(fixture.info.images.size(), 1U);
        EXPECT_EQ(fixture.info.images.front().dynamic_descriptor_count, NUM_IMAGES);
        EXPECT_EQ(fixture.info.samplers.size(), 1U);
        EXPECT_EQ(std::ranges::count_if(fixture.block->Instructions(), [](const IR::Inst& inst) {
                      return inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLodDynamic;
                  }),
                  0U);
    }
}

TEST(DynamicImageTableResourceTracking,
     RuntimeDescriptorChangesSelectDistinctAndRepeatableSpecializations) {
    DynamicImageTableFixture fixture;
    IR::Inst* const access = fixture.MakePatchableAccess(0U, false, false);
    const auto match = MatchDynamicImageTable(*access);
    ASSERT_TRUE(match.has_value());
    constexpr u32 Destination = 20U;
    InstallDynamicImageTable(fixture, *match, Destination);
    const auto profile = DynamicImageProfile();
    ResourceTrackingPass(fixture.program, profile);
    ASSERT_EQ(fixture.info.images.size(), 1U);

    const RuntimeInfo runtime_info{};
    const Backend::Bindings bindings{};
    const StageSpecialization supported_first{fixture.info, runtime_info, profile, bindings};

    auto unsupported_model = fixture.info.images.front().ReadSharpUnchecked(fixture.info);
    unsupported_model.type = static_cast<u64>(AmdGpu::ImageType::Color2DMsaa);
    std::memcpy(fixture.info.flattened_ud_buf.data() + Destination, &unsupported_model,
                sizeof(AmdGpu::Image));
    const StageSpecialization unsupported{fixture.info, runtime_info, profile, bindings};

    unsupported_model.type = static_cast<u64>(AmdGpu::ImageType::Color2D);
    std::memcpy(fixture.info.flattened_ud_buf.data() + Destination, &unsupported_model,
                sizeof(AmdGpu::Image));
    const StageSpecialization supported_again{fixture.info, runtime_info, profile, bindings};

    EXPECT_FALSE(supported_first == unsupported);
    EXPECT_FALSE(unsupported == supported_first);
    EXPECT_TRUE(supported_first == supported_again);
    EXPECT_TRUE(supported_again == supported_first);
}

TEST(DynamicImageTablePermutation, InvalidAndValidModelsRemainDistinctInEitherEncounterOrder) {
    constexpr u32 Destination = 20U;
    const auto profile = DynamicImageProfile();
    const RuntimeInfo runtime_info{};
    const Backend::Bindings bindings{};

    DynamicImageTableFixture invalid_fixture;
    IR::Inst* const invalid_access = invalid_fixture.MakePatchableAccess(0U, false, false);
    const auto invalid_match = MatchDynamicImageTable(*invalid_access);
    ASSERT_TRUE(invalid_match.has_value());
    InstallDynamicImageTable(invalid_fixture, *invalid_match, Destination, AmdGpu::Image{});
    ResourceTrackingPass(invalid_fixture.program, profile);
    EXPECT_EQ(std::ranges::count_if(
                  invalid_fixture.block->Instructions(), [](const IR::Inst& inst) {
                      return inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLodDynamic;
                  }),
              0U);
    const StageSpecialization invalid_first{invalid_fixture.info, runtime_info, profile, bindings};

    DynamicImageTableFixture valid_fixture;
    IR::Inst* const valid_access = valid_fixture.MakePatchableAccess(0U, false, false);
    const auto valid_match = MatchDynamicImageTable(*valid_access);
    ASSERT_TRUE(valid_match.has_value());
    InstallDynamicImageTable(valid_fixture, *valid_match, Destination);
    ResourceTrackingPass(valid_fixture.program, profile);
    EXPECT_EQ(std::ranges::count_if(
                  valid_fixture.block->Instructions(), [](const IR::Inst& inst) {
                      return inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLodDynamic;
                  }),
              1U);
    const StageSpecialization valid_later{valid_fixture.info, runtime_info, profile, bindings};

    ASSERT_EQ(invalid_first.images.size(), 1U);
    ASSERT_EQ(valid_later.images.size(), 1U);
    const auto& invalid_image = invalid_first.images.front();
    const auto& valid_image = valid_later.images.front();
    EXPECT_EQ(invalid_image.type, valid_image.type);
    EXPECT_EQ(invalid_image.number_class, valid_image.number_class);
    EXPECT_EQ(invalid_image.is_storage, valid_image.is_storage);
    EXPECT_EQ(invalid_image.is_cube, valid_image.is_cube);
    EXPECT_EQ(invalid_image.is_srgb, valid_image.is_srgb);
    EXPECT_EQ(invalid_image.dst_select, valid_image.dst_select);
    EXPECT_EQ(invalid_image.num_conversion, valid_image.num_conversion);
    EXPECT_EQ(invalid_image.num_bindings, valid_image.num_bindings);
    EXPECT_FALSE(invalid_image.has_supported_dynamic_model);
    EXPECT_TRUE(valid_image.has_supported_dynamic_model);
    auto normalized_invalid_image = invalid_image;
    normalized_invalid_image.has_supported_dynamic_model = valid_image.has_supported_dynamic_model;
    EXPECT_EQ(normalized_invalid_image, valid_image);

    EXPECT_FALSE(invalid_first == valid_later);
    EXPECT_FALSE(valid_later == invalid_first);

    constexpr u64 InvalidModuleIdentity = 0xBAD0U;
    constexpr u64 ValidModuleIdentity = 0x600DU;
    struct SelectionResult {
        size_t module_count{};
        std::array<u64, 3> identities{};
    };
    const auto select_in_order = [&](bool invalid_first_encounter) {
        Vulkan::Program program{};
        const auto select = [&](StageSpecialization spec, u64 identity) {
            auto it = program.FindPermut(spec);
            if (it == program.modules.end()) {
                Info compiled_info{};
                compiled_info.pgm_hash = identity;
                program.AddPermut(vk::ShaderModule{}, std::move(spec), std::move(compiled_info));
                it = std::prev(program.modules.end());
            }
            return it->info->pgm_hash;
        };

        SelectionResult result{};
        if (invalid_first_encounter) {
            result.identities = {
                select(invalid_first, InvalidModuleIdentity),
                select(valid_later, ValidModuleIdentity),
                select(invalid_first, InvalidModuleIdentity),
            };
        } else {
            result.identities = {
                select(valid_later, ValidModuleIdentity),
                select(invalid_first, InvalidModuleIdentity),
                select(valid_later, ValidModuleIdentity),
            };
        }
        result.module_count = program.modules.size();
        return result;
    };

    const auto invalid_valid_invalid = select_in_order(true);
    EXPECT_EQ(invalid_valid_invalid.module_count, 2U);
    EXPECT_EQ(invalid_valid_invalid.identities,
              (std::array<u64, 3>{InvalidModuleIdentity, ValidModuleIdentity,
                                  InvalidModuleIdentity}));

    const auto valid_invalid_valid = select_in_order(false);
    EXPECT_EQ(valid_invalid_valid.module_count, 2U);
    EXPECT_EQ(valid_invalid_valid.identities,
              (std::array<u64, 3>{ValidModuleIdentity, InvalidModuleIdentity,
                                  ValidModuleIdentity}));
}

TEST(DynamicImageTablePermutation, SupportedModelFlagRoundTripsWithImageSpecializations) {
    decltype(StageSpecialization{}.images) original;
    original.push_back({.has_supported_dynamic_model = false, .num_bindings = 64U});
    original.push_back({.has_supported_dynamic_model = true, .num_bindings = 32U});

    Serialization::Archive output;
    Serialization::Writer writer{output};
    writer.Write(original);

    Serialization::Archive input{output.TakeOff()};
    decltype(StageSpecialization{}.images) restored;
    Serialization::Reader reader{input};
    reader.Read(restored);

    EXPECT_EQ(restored, original);
    ASSERT_EQ(restored.size(), 2U);
    EXPECT_FALSE(restored[0].has_supported_dynamic_model);
    EXPECT_TRUE(restored[1].has_supported_dynamic_model);
    EXPECT_TRUE(input.IsEoS());
}

TEST(DynamicImageTableSpirv, UnsupportedFirstMsaaArrayDescriptorDoesNotCrashEmission) {
    DynamicImageTableFixture fixture;
    IR::Inst* const access = fixture.MakePatchableAccess(0U, true);
    const auto match = MatchDynamicImageTable(*access);
    ASSERT_TRUE(match.has_value());
    auto msaa_array = AmdGpu::Image::Null(false);
    msaa_array.type = static_cast<u64>(AmdGpu::ImageType::Color2DMsaaArray);
    InstallDynamicImageTable(fixture, *match, 20U, msaa_array);

    auto profile = DynamicImageProfile();
    profile.supported_spirv = 0x00010600U;
    profile.subgroup_size = 32U;
    ResourceTrackingPass(fixture.program, profile);
    ASSERT_EQ(fixture.info.images.size(), 1U);
    EXPECT_TRUE(fixture.info.images.front().is_array);
    EXPECT_EQ(std::ranges::count_if(fixture.block->Instructions(), [](const IR::Inst& inst) {
                  return inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLodDynamic;
              }),
              0U);

    fixture.emitter->Epilogue();
    fixture.program.syntax_list.emplace_back();
    fixture.program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    fixture.program.syntax_list.back().data.block = fixture.block;
    fixture.program.syntax_list.emplace_back();
    fixture.program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    fixture.program.post_order_blocks = IR::PostOrder(fixture.program.syntax_list.front());
    DeadCodeEliminationPass(fixture.program);
    CollectShaderInfoPass(fixture.program, profile);

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.cs_info.workgroup_size = {1U, 1U, 1U};
    Backend::Bindings bindings{};
    const auto spirv = Backend::SPIRV::EmitSPIRV(profile, runtime_info, fixture.program, bindings);

    ASSERT_GE(spirv.size(), 5U);
    EXPECT_EQ(spirv.front(), 0x07230203U);
}

TEST(DynamicImageTableSpirv,
     EmitsAValidatedFb974Shadow256BitGuardedNonuniform64DescriptorAccess) {
    DynamicImageTableFixture fixture;
    IR::Inst* const access = fixture.MakePatchableAccess(0U, false, false);
    const auto match = MatchDynamicImageTable(*access);
    ASSERT_TRUE(match.has_value());
    InstallDynamicImageTable(fixture, *match, 20U);

    auto profile = DynamicImageProfile();
    profile.supported_spirv = 0x00010600U;
    profile.subgroup_size = 32U;
    ResourceTrackingPass(fixture.program, profile);
    ASSERT_EQ(fixture.info.images.size(), 1U);
    EXPECT_FALSE(fixture.info.images.front().is_r128);
    EXPECT_EQ(fixture.info.images.front().dynamic_descriptor_count, NUM_IMAGES);
    EXPECT_EQ(fixture.info.images.front().dynamic_descriptor_stride_dw, 8U);
    fixture.emitter->Epilogue();
    fixture.program.syntax_list.emplace_back();
    fixture.program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    fixture.program.syntax_list.back().data.block = fixture.block;
    fixture.program.syntax_list.emplace_back();
    fixture.program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    fixture.program.post_order_blocks = IR::PostOrder(fixture.program.syntax_list.front());
    DeadCodeEliminationPass(fixture.program);
    CollectShaderInfoPass(fixture.program, profile);

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.cs_info.workgroup_size = {1U, 1U, 1U};
    Backend::Bindings bindings{};
    const auto spirv = Backend::SPIRV::EmitSPIRV(profile, runtime_info, fixture.program, bindings);

    ASSERT_GE(spirv.size(), 5U);
    EXPECT_EQ(spirv.front(), 0x07230203U);
    EXPECT_TRUE(HasSpirvCapability(spirv, spv::CapabilityShaderNonUniform));
    EXPECT_TRUE(
        HasSpirvCapability(spirv, spv::CapabilitySampledImageArrayNonUniformIndexing));
    EXPECT_TRUE(HasSpirvArrayLength(spirv, NUM_IMAGES));
    EXPECT_GE(CountSpirvOpcode(spirv, spv::OpSelectionMerge), 1U);
    EXPECT_GE(CountSpirvOpcode(spirv, spv::OpAccessChain), 1U);
    EXPECT_GE(CountSpirvOpcode(spirv, spv::OpPhi), 1U);

#ifdef SHADPS4_SPIRV_VAL_PATH
    const auto output =
        std::filesystem::temp_directory_path() / "shadps4-dynamic-image-table-test.spv";
    {
        std::ofstream file{output, std::ios::binary | std::ios::trunc};
        ASSERT_TRUE(file.good());
        file.write(reinterpret_cast<const char*>(spirv.data()),
                   static_cast<std::streamsize>(spirv.size() * sizeof(u32)));
    }
    EXPECT_EQ(RunSpirvValidator(output), 0);
    std::error_code remove_error;
    std::filesystem::remove(output, remove_error);
#endif
}

} // namespace
} // namespace Shader::Optimization
