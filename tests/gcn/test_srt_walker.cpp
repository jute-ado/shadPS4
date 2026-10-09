// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "common/serdes.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/recompiler.h"

namespace Shader {
namespace {

struct WalkerFixture {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    std::array<u32, 8> root{};
    std::array<u32, 132> stale_child{};
    std::array<u32, 132> fresh_child{};

    WalkerFixture() {
        info.stage = Stage::Vertex;
        info.l_stage = LogicalStage::Vertex;
        info.user_data = user_data;

        const VAddr root_address = reinterpret_cast<VAddr>(root.data());
        user_data[0] = static_cast<u32>(root_address);
        user_data[1] = static_cast<u32>(root_address >> 32);

        auto* const block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        program.post_order_blocks.push_back(block);

        IR::IREmitter ir{*block};
        const auto root_pointer = ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0),
                                                        ir.GetUserData(IR::ScalarReg::S1));
        const auto child_lo = ir.ReadConst(root_pointer, ir.Imm32(2U));
        const auto child_hi = ir.ReadConst(root_pointer, ir.Imm32(3U));
        const auto child_pointer = ir.CompositeConstruct(child_lo, child_hi);
        ir.Reference(ir.ReadConst(child_pointer, ir.Imm32(130U)));
        ir.Reference(ir.ReadConst(child_pointer, ir.Imm32(131U)));
    }

    void PointRootAt(std::array<u32, 132>& child) {
        const VAddr address = reinterpret_cast<VAddr>(child.data());
        root[2] = static_cast<u32>(address);
        root[3] = static_cast<u32>(address >> 32);
    }
};

struct WalkerSyncRecorder {
    WalkerFixture* fixture{};
    std::vector<std::pair<VAddr, u64>> ranges;
    u32 fresh_x{};
    u32 fresh_y{};
    bool reject_root_pointer{};
    bool reject_child_data{};
    bool publish_null_child{};
};

struct TableReadRecorder {
    std::vector<std::pair<VAddr, u64>> ranges;
    VAddr rejected_address{};
};

bool PS4_SYSV_ABI ReadPrivateRange(void* opaque, VAddr address, void* output, u64 size) {
    auto& recorder = *static_cast<TableReadRecorder*>(opaque);
    recorder.ranges.emplace_back(address, size);
    if (address == recorder.rejected_address) {
        return false;
    }
    std::memcpy(output, reinterpret_cast<const void*>(address), size);
    return true;
}

bool PS4_SYSV_ABI ReadWalkerRange(void* opaque, VAddr address, void* output, u64 size) {
    auto& recorder = *static_cast<WalkerSyncRecorder*>(opaque);
    auto& fixture = *recorder.fixture;
    recorder.ranges.emplace_back(address, size);

    const VAddr root_pointer_address = reinterpret_cast<VAddr>(fixture.root.data() + 2);
    if (address == root_pointer_address && size == 8) {
        if (recorder.reject_root_pointer) {
            return false;
        }
        if (recorder.publish_null_child) {
            std::memset(output, 0, size);
            return true;
        }
        const VAddr fresh_address = reinterpret_cast<VAddr>(fixture.fresh_child.data());
        std::memcpy(output, &fresh_address, sizeof(fresh_address));
        return true;
    }

    const VAddr child_data_address = reinterpret_cast<VAddr>(fixture.fresh_child.data() + 130);
    if (address == child_data_address && size == 8) {
        if (recorder.reject_child_data) {
            return false;
        }
        const std::array fresh_values{recorder.fresh_x, recorder.fresh_y};
        std::memcpy(output, fresh_values.data(), sizeof(fresh_values));
        return true;
    }
    return false;
}

bool PS4_SYSV_ABI RejectWalkerRange(void*, VAddr, void*, u64) {
    return false;
}

TEST(SrtWalker, SynchronizesParentBeforeNestedEightBytePointerLoad) {
    WalkerFixture fixture;
    fixture.PointRootAt(fixture.stale_child);
    fixture.stale_child[130] = 250U;

    WalkerSyncRecorder recorder{
        .fixture = &fixture,
        .fresh_x = std::bit_cast<u32>(1.0F / 1920.0F),
        .fresh_y = std::bit_cast<u32>(1.0F / 1080.0F),
    };
    const SrtMemoryReader reader{ReadWalkerRange, &recorder};

    Optimization::FlattenExtendedUserdataPass(fixture.program, reader);

    ASSERT_EQ(recorder.ranges.size(), 2U);
    EXPECT_EQ(recorder.ranges[0],
              std::make_pair(reinterpret_cast<VAddr>(fixture.root.data() + 2), 8ULL));
    EXPECT_EQ(recorder.ranges[1],
              std::make_pair(reinterpret_cast<VAddr>(fixture.fresh_child.data() + 130), 8ULL));
    ASSERT_GE(fixture.info.flattened_ud_buf.size(), 20U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[18], recorder.fresh_x);
    EXPECT_EQ(fixture.info.flattened_ud_buf[19], recorder.fresh_y);
    EXPECT_EQ(fixture.root[2],
              static_cast<u32>(reinterpret_cast<VAddr>(fixture.stale_child.data())));
    EXPECT_EQ(fixture.stale_child[130], 250U);
}

TEST(SrtWalker, UsesSameSynchronizerForInitialAndCachedRefresh) {
    WalkerFixture fixture;
    fixture.PointRootAt(fixture.stale_child);

    WalkerSyncRecorder recorder{
        .fixture = &fixture,
        .fresh_x = 0x3A112233U,
        .fresh_y = 0x3A445566U,
    };
    const SrtMemoryReader reader{ReadWalkerRange, &recorder};

    Optimization::FlattenExtendedUserdataPass(fixture.program, reader);
    ASSERT_EQ(recorder.ranges.size(), 2U);

    fixture.PointRootAt(fixture.stale_child);
    recorder.ranges.clear();
    fixture.info.RefreshFlatBuf(reader);

    ASSERT_EQ(recorder.ranges.size(), 2U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[18], recorder.fresh_x);
    EXPECT_EQ(fixture.info.flattened_ud_buf[19], recorder.fresh_y);
}

TEST(SrtWalker, RefusedParentRangeDoesNotPublishStaleNestedData) {
    WalkerFixture fixture;
    fixture.PointRootAt(fixture.stale_child);
    fixture.stale_child[130] = 0xDEADBEEFU;
    fixture.stale_child[131] = 0xBAD0CAFEU;

    WalkerSyncRecorder recorder{.fixture = &fixture, .reject_root_pointer = true};
    const SrtMemoryReader reader{ReadWalkerRange, &recorder};

    Optimization::FlattenExtendedUserdataPass(fixture.program, reader);

    ASSERT_EQ(recorder.ranges.size(), 1U);
    EXPECT_EQ(recorder.ranges[0],
              std::make_pair(reinterpret_cast<VAddr>(fixture.root.data() + 2), 8ULL));
    ASSERT_GE(fixture.info.flattened_ud_buf.size(), 20U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[16], 0U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[17], 0U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[18], 0U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[19], 0U);
}

TEST(SrtWalker, SuccessfullyReadNullNestedPointerRemainsAValidSparseGeneration) {
    WalkerFixture fixture;
    WalkerSyncRecorder recorder{.fixture = &fixture, .publish_null_child = true};
    CompileResourceCapture capture{{ReadWalkerRange, &recorder}};

    Optimization::FlattenExtendedUserdataPass(fixture.program, capture.Reader());
    const auto snapshot = capture.Finish(fixture.info);

    ASSERT_EQ(recorder.ranges.size(), 1U);
    EXPECT_EQ(recorder.ranges[0],
              std::make_pair(reinterpret_cast<VAddr>(fixture.root.data() + 2), 8ULL));
    ASSERT_GE(fixture.info.flattened_ud_buf.size(), 20U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[16], 0U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[17], 0U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[18], 0U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[19], 0U);
    EXPECT_TRUE(snapshot.valid);
}

TEST(SrtWalker, RefusedLeafRangePublishesZerosInsteadOfStaleGuestData) {
    WalkerFixture fixture;
    fixture.PointRootAt(fixture.stale_child);
    fixture.stale_child[130] = 0xDEADBEEFU;
    fixture.stale_child[131] = 0xBAD0CAFEU;

    WalkerSyncRecorder recorder{.fixture = &fixture, .reject_child_data = true};
    const SrtMemoryReader reader{ReadWalkerRange, &recorder};

    Optimization::FlattenExtendedUserdataPass(fixture.program, reader);

    ASSERT_EQ(recorder.ranges.size(), 2U);
    ASSERT_GE(fixture.info.flattened_ud_buf.size(), 20U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[18], 0U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[19], 0U);
}

TEST(SrtWalker, LargeContiguousNodeHandlesRefusedRange) {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    std::array<u32, 96> root{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.user_data = user_data;
    const VAddr root_address = reinterpret_cast<VAddr>(root.data());
    user_data[0] = static_cast<u32>(root_address);
    user_data[1] = static_cast<u32>(root_address >> 32);

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);
    IR::IREmitter ir{*block};
    const auto root_pointer =
        ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0), ir.GetUserData(IR::ScalarReg::S1));
    for (u32 i = 0; i < 64; ++i) {
        root[i] = 0xA0000000U + i;
        ir.Reference(ir.ReadConst(root_pointer, ir.Imm32(i)));
    }

    const SrtMemoryReader reader{RejectWalkerRange, nullptr};
    EXPECT_NO_THROW(Optimization::FlattenExtendedUserdataPass(program, reader));
    ASSERT_GE(info.flattened_ud_buf.size(), NUM_USER_DATA_REGS + 64U);
    for (u32 i = 0; i < 64; ++i) {
        EXPECT_EQ(info.flattened_ud_buf[NUM_USER_DATA_REGS + i], 0U);
    }
}

TEST(SrtWalker, SnapshotsFb974Shadow256BitAffineDynamicImageTableAndReplaysEveryDword) {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    std::array<u32, 8> root{};
    std::array<u32, 4200> table_storage{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.user_data = user_data;
    const VAddr root_address = reinterpret_cast<VAddr>(root.data());
    const VAddr table_address = reinterpret_cast<VAddr>(table_storage.data());
    user_data[0] = static_cast<u32>(root_address);
    user_data[1] = static_cast<u32>(root_address >> 32);
    root[2] = static_cast<u32>(table_address);
    root[3] = static_cast<u32>(table_address >> 32);

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);
    IR::IREmitter ir{*block};
    const auto root_pointer =
        ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0), ir.GetUserData(IR::ScalarReg::S1));
    // Retain the captured layout: two earlier root values occupy flat 16/17, so the affine table
    // base pair is flattened to 18/19.
    ir.Reference(ir.ReadConst(root_pointer, ir.Imm32(0U)));
    ir.Reference(ir.ReadConst(root_pointer, ir.Imm32(1U)));
    const auto table_pointer = ir.CompositeConstruct(ir.ReadConst(root_pointer, ir.Imm32(2U)),
                                                     ir.ReadConst(root_pointer, ir.Imm32(3U)));
    // Accepted fb974 at PC 0xe60: affine byte offset 14480, byte stride 32, R128=0.
    const auto byte_offset = ir.IAdd(ir.IMul(ir.LaneId(), ir.Imm32(32U)), ir.Imm32(14480U));
    const auto image = ir.ReadConst(table_pointer, ir.ShiftRightLogical(byte_offset, ir.Imm32(2U)));
    IR::TextureInstInfo texture_info{};
    texture_info.has_lod.Assign(true);
    texture_info.is_r128.Assign(false);
    const auto sample = ir.ImageSampleRaw(image, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U),
                                          ir.Imm32(0U), ir.Imm32(0U), texture_info);
    ir.Reference(sample);

    constexpr u32 table_offset_dw = 14480U / sizeof(u32);
    constexpr u32 table_stride_dw = 32U / sizeof(u32);
    constexpr u32 table_span_dw = (NUM_IMAGES - 1U) * table_stride_dw + 8U;
    static_assert(table_span_dw == 512U);
    for (u32 i = 0; i < table_span_dw; ++i) {
        table_storage[table_offset_dw + i] = 0xA5000000U + i;
    }
    TableReadRecorder recorder{};
    const SrtMemoryReader reader{ReadPrivateRange, &recorder};

    Optimization::FlattenExtendedUserdataPass(program, reader);

    ASSERT_EQ(info.srt_info.range_reservations.size(), 1U);
    const auto& reservation = info.srt_info.range_reservations.front();
    EXPECT_EQ(reservation.pointer_lo_flat_index, 18U);
    EXPECT_EQ(reservation.source_offset_dw, table_offset_dw);
    EXPECT_EQ(reservation.destination_flat_offset_dw, 20U);
    EXPECT_EQ(reservation.span_dw, table_span_dw);
    EXPECT_EQ(image.InstRecursive()->Flags<u32>(), reservation.destination_flat_offset_dw);
    ASSERT_GE(info.flattened_ud_buf.size(), reservation.destination_flat_offset_dw + table_span_dw);
    EXPECT_EQ(info.flattened_ud_buf[reservation.destination_flat_offset_dw], 0xA5000000U);
    EXPECT_EQ(info.flattened_ud_buf[reservation.destination_flat_offset_dw + table_span_dw - 1],
              0xA5000000U + table_span_dw - 1);

    table_storage[table_offset_dw] = 0x11223344U;
    table_storage[table_offset_dw + table_span_dw - 1U] = 0x55667788U;
    recorder.ranges.clear();
    info.RefreshFlatBuf(reader);
    EXPECT_EQ(info.flattened_ud_buf[reservation.destination_flat_offset_dw], 0x11223344U);
    EXPECT_EQ(info.flattened_ud_buf[reservation.destination_flat_offset_dw + 511U], 0x55667788U);
    ASSERT_EQ(recorder.ranges.size(), 2U);
    EXPECT_EQ(recorder.ranges[1],
              std::make_pair(table_address + 14480U, u64{table_span_dw * sizeof(u32)}));
}

TEST(SrtWalker, RefusedDynamicImageTableSnapshotZeroFillsEveryReservedDword) {
    Info info{};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    std::array<u32, 32> table{};
    info.user_data = user_data;
    const VAddr table_address = reinterpret_cast<VAddr>(table.data());
    user_data[4] = static_cast<u32>(table_address);
    user_data[5] = static_cast<u32>(table_address >> 32);
    info.srt_info.range_reservations.push_back({
        .pointer_lo_flat_index = 4U,
        .source_offset_dw = 3U,
        .destination_flat_offset_dw = NUM_USER_DATA_REGS,
        .span_dw = 8U,
    });
    info.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS + 8U;
    info.flattened_ud_buf.assign(info.srt_info.flattened_bufsize_dw, 0xDEADBEEFU);
    TableReadRecorder recorder{.rejected_address = table_address + 3U * sizeof(u32)};
    const SrtMemoryReader reader{ReadPrivateRange, &recorder};

    info.RefreshFlatBuf(reader);

    ASSERT_EQ(info.flattened_ud_buf.size(), NUM_USER_DATA_REGS + 8U);
    for (u32 i = NUM_USER_DATA_REGS; i < info.flattened_ud_buf.size(); ++i) {
        EXPECT_EQ(info.flattened_ud_buf[i], 0U);
    }
}

TEST(SrtWalker, MissingReaderNeverFallsBackToRawGuestTableDereference) {
    Info info{};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    std::array<u32, 8> table{};
    info.user_data = user_data;
    const VAddr table_address = reinterpret_cast<VAddr>(table.data());
    user_data[6] = static_cast<u32>(table_address);
    user_data[7] = static_cast<u32>(table_address >> 32);
    std::ranges::fill(table, 0xCAFEBABEU);
    info.srt_info.range_reservations.push_back({
        .pointer_lo_flat_index = 6U,
        .source_offset_dw = 0U,
        .destination_flat_offset_dw = NUM_USER_DATA_REGS,
        .span_dw = static_cast<u32>(table.size()),
    });
    info.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS + table.size();

    info.RefreshFlatBuf();

    for (u32 i = NUM_USER_DATA_REGS; i < info.flattened_ud_buf.size(); ++i) {
        EXPECT_EQ(info.flattened_ud_buf[i], 0U);
    }
}

TEST(SrtMetadata, RoundTripsRangeReservationsWithoutSerializingContainerInternals) {
    PersistentSrtInfo original{};
    original.flattened_bufsize_dw = 530U;
    original.range_reservations.push_back({
        .pointer_lo_flat_index = 16U,
        .source_offset_dw = 3620U,
        .destination_flat_offset_dw = 18U,
        .span_dw = 512U,
    });
    Serialization::Archive output;
    original.Serialize(output);
    Serialization::Archive input{output.TakeOff()};
    PersistentSrtInfo restored{};

    ASSERT_TRUE(restored.Deserialize(input));

    EXPECT_EQ(restored.walker_func, nullptr);
    EXPECT_EQ(restored.walker_func_size, 0U);
    EXPECT_EQ(restored.flattened_bufsize_dw, original.flattened_bufsize_dw);
    EXPECT_EQ(restored.range_reservations, original.range_reservations);
    ASSERT_EQ(restored.range_reservations.size(), 1U);
    EXPECT_EQ(restored.range_reservations.front().source_offset_dw, 3620U);
    EXPECT_EQ(restored.range_reservations.front().destination_flat_offset_dw, 18U);
    EXPECT_EQ(restored.range_reservations.front().span_dw, 512U);
    EXPECT_TRUE(input.IsEoS());
}

} // namespace
} // namespace Shader
