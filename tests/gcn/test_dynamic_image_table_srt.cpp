// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <limits>

#include <gtest/gtest.h>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/recompiler.h"

namespace Shader::Optimization {
namespace {

struct DynamicImageTableSrtFixture {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    std::array<u32, 8> root{};
    std::array<u32, 4200> table{};
    IR::Value image_handle{};

    explicit DynamicImageTableSrtFixture(bool explicit_lod = true) {
        info.stage = Stage::Compute;
        info.l_stage = LogicalStage::Compute;
        info.user_data = user_data;

        const VAddr root_address = reinterpret_cast<VAddr>(root.data());
        const VAddr table_address = reinterpret_cast<VAddr>(table.data());
        user_data[0] = static_cast<u32>(root_address);
        user_data[1] = static_cast<u32>(root_address >> 32);
        root[2] = static_cast<u32>(table_address);
        root[3] = static_cast<u32>(table_address >> 32);

        auto* const block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        program.post_order_blocks.push_back(block);
        IR::IREmitter ir{*block};
        const auto root_pointer = ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0),
                                                        ir.GetUserData(IR::ScalarReg::S1));
        // These two ordinary values make the table pointer land at flat indices 18 and 19.
        ir.Reference(ir.ReadConst(root_pointer, ir.Imm32(0U)));
        ir.Reference(ir.ReadConst(root_pointer, ir.Imm32(1U)));
        const auto table_pointer = ir.CompositeConstruct(ir.ReadConst(root_pointer, ir.Imm32(2U)),
                                                         ir.ReadConst(root_pointer, ir.Imm32(3U)));
        const auto byte_offset = ir.IAdd(ir.IMul(ir.LaneId(), ir.Imm32(32U)), ir.Imm32(14480U));
        image_handle = ir.ReadConst(table_pointer, ir.ShiftRightLogical(byte_offset, ir.Imm32(2U)));
        IR::TextureInstInfo texture_info{};
        texture_info.has_lod.Assign(explicit_lod);
        texture_info.is_r128.Assign(false);
        ir.Reference(ir.ImageSampleRaw(image_handle, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U),
                                       ir.Imm32(0U), ir.Imm32(0U), texture_info));
    }
};

TEST(DynamicImageTableSrt, ReservesTheProvenSparseWindowWithoutActivatingTheHandle) {
    DynamicImageTableSrtFixture fixture;

    EXPECT_FALSE(FlattenExtendedUserdataPass(fixture.program));

    constexpr u32 table_offset_dw = 14480U / sizeof(u32);
    constexpr u32 table_stride_dw = 32U / sizeof(u32);
    constexpr u32 table_span_dw = (NUM_IMAGES - 1U) * table_stride_dw + 8U;
    static_assert(table_span_dw == 512U);
    ASSERT_EQ(fixture.info.srt_info.range_reservations.size(), 1U);
    const auto& reservation = fixture.info.srt_info.range_reservations.front();
    EXPECT_EQ(reservation.pointer_lo_flat_index, 18U);
    EXPECT_EQ(reservation.source_offset_dw, table_offset_dw);
    EXPECT_EQ(reservation.destination_flat_offset_dw, 20U);
    EXPECT_EQ(reservation.span_dw, table_span_dw);
    EXPECT_EQ(fixture.image_handle.InstRecursive()->Flags<u32>(), 0U);
    EXPECT_EQ(fixture.info.srt_info.flattened_bufsize_dw,
              reservation.destination_flat_offset_dw + reservation.span_dw);
}

TEST(DynamicImageTableSrt, CoalescesSortedRangesWithoutInvalidatingEarlierDestinations) {
    PersistentSrtInfo srt{};
    srt.flattened_bufsize_dw = 20U;

    const auto first = srt.TryReserveRange(18U, 100U, 8U);
    const auto contained = srt.TryReserveRange(18U, 104U, 2U);
    const auto adjacent = srt.TryReserveRange(18U, 108U, 4U);
    const auto disjoint = srt.TryReserveRange(18U, 200U, 4U);

    ASSERT_EQ(first, 20U);
    EXPECT_EQ(contained, 24U);
    EXPECT_EQ(adjacent, 28U);
    EXPECT_EQ(disjoint, 32U);
    EXPECT_EQ(srt.FindReservedRange(18U, 102U, 4U), 22U);
    EXPECT_FALSE(srt.FindReservedRange(18U, 110U, 4U).has_value());
    ASSERT_EQ(srt.range_reservations.size(), 2U);
    EXPECT_EQ(srt.range_reservations[0].span_dw, 12U);
    EXPECT_EQ(srt.flattened_bufsize_dw, 36U);
    EXPECT_TRUE(srt.ValidateRangeReservations());
}

TEST(DynamicImageTableSrt, MalformedOrNondeterministicRangesFailWithoutMutation) {
    PersistentSrtInfo srt{};
    srt.flattened_bufsize_dw = 20U;
    ASSERT_EQ(srt.TryReserveRange(18U, 100U, 8U), 20U);
    const auto original_ranges = srt.range_reservations;
    const u32 original_size = srt.flattened_bufsize_dw;

    EXPECT_FALSE(srt.TryReserveRange(19U, 200U, 0U).has_value());
    EXPECT_FALSE(srt.TryReserveRange(19U, std::numeric_limits<u32>::max(), 2U).has_value());
    EXPECT_FALSE(srt.TryReserveRange(22U, 300U, 4U).has_value());
    EXPECT_FALSE(srt.TryReserveRange(18U, 98U, 4U).has_value());
    EXPECT_FALSE(srt.TryReserveRange(17U, 300U, 4U).has_value());
    EXPECT_EQ(srt.range_reservations, original_ranges);
    EXPECT_EQ(srt.flattened_bufsize_dw, original_size);

    PersistentSrtInfo overflowing{};
    overflowing.flattened_bufsize_dw = std::numeric_limits<u32>::max();
    EXPECT_FALSE(overflowing.TryReserveRange(0U, 0U, 1U).has_value());
    EXPECT_TRUE(overflowing.range_reservations.empty());

    PersistentSrtInfo oversized{};
    oversized.flattened_bufsize_dw = PersistentSrtInfo::MaxRangeSnapshotDwords;
    EXPECT_FALSE(oversized.TryReserveRange(0U, 0U, 1U).has_value());
}

TEST(DynamicImageTableSrt, AdjacentExtensionCannotGrowPastTheSnapshotCap) {
    PersistentSrtInfo srt{};
    constexpr u32 first_span = PersistentSrtInfo::MaxRangeSnapshotDwords - NUM_USER_DATA_REGS;
    ASSERT_EQ(srt.TryReserveRange(0U, 0U, first_span), NUM_USER_DATA_REGS);
    ASSERT_EQ(srt.flattened_bufsize_dw, PersistentSrtInfo::MaxRangeSnapshotDwords);
    const auto original_ranges = srt.range_reservations;
    const u32 original_size = srt.flattened_bufsize_dw;

    EXPECT_FALSE(srt.TryReserveRange(0U, first_span, 1U).has_value());
    EXPECT_EQ(srt.range_reservations, original_ranges);
    EXPECT_EQ(srt.flattened_bufsize_dw, original_size);

    PersistentSrtInfo malformed{};
    malformed.flattened_bufsize_dw = PersistentSrtInfo::MaxRangeSnapshotDwords + 1U;
    EXPECT_FALSE(malformed.ValidateRangeReservations());
}

TEST(DynamicImageTableSrt, DoesNotReserveImplicitLodAccesses) {
    DynamicImageTableSrtFixture fixture{false};

    EXPECT_TRUE(FlattenExtendedUserdataPass(fixture.program));

    EXPECT_TRUE(fixture.info.srt_info.range_reservations.empty());
    EXPECT_EQ(fixture.info.srt_info.flattened_bufsize_dw, 20U);
}

} // namespace
} // namespace Shader::Optimization
