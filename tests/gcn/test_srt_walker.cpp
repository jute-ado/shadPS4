// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <ranges>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

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
    bool failed{};
};

bool PS4_SYSV_ABI ReadWalkerRange(void* opaque, VAddr address, void* output, u64 size) {
    auto& recorder = *static_cast<WalkerSyncRecorder*>(opaque);
    auto& fixture = *recorder.fixture;
    recorder.ranges.emplace_back(address, size);

    const VAddr root_pointer_address = reinterpret_cast<VAddr>(fixture.root.data() + 2);
    if (address == root_pointer_address && size == 8) {
        if (recorder.reject_root_pointer) {
            recorder.failed = true;
            return false;
        }
        const VAddr fresh_child = reinterpret_cast<VAddr>(fixture.fresh_child.data());
        std::memcpy(output, &fresh_child, sizeof(fresh_child));
        return true;
    }

    const VAddr child_data_address = reinterpret_cast<VAddr>(fixture.fresh_child.data() + 130);
    if (address == child_data_address && size == 8) {
        if (recorder.reject_child_data) {
            recorder.failed = true;
            return false;
        }
        const std::array values{recorder.fresh_x, recorder.fresh_y};
        std::memcpy(output, values.data(), sizeof(values));
        return true;
    }
    recorder.failed = true;
    return false;
}

bool PS4_SYSV_ABI RejectWalkerRange(void* opaque, VAddr, void*, u64) {
    *static_cast<bool*>(opaque) = true;
    return false;
}

struct ReservationSource {
    VAddr address{};
    const void* data{};
    u64 size{};
};

struct ReservationReader {
    std::vector<ReservationSource> sources;
    std::vector<std::pair<VAddr, u64>> ranges;
    size_t reject_call{std::numeric_limits<size_t>::max()};
    bool failed{};
};

bool PS4_SYSV_ABI ReadReservationRange(void* opaque, VAddr address, void* output, u64 size) {
    auto& reader = *static_cast<ReservationReader*>(opaque);
    const size_t call = reader.ranges.size();
    reader.ranges.emplace_back(address, size);
    if (call == reader.reject_call) {
        reader.failed = true;
        return false;
    }
    for (const auto& source : reader.sources) {
        if (source.address == address && source.size == size) {
            std::memcpy(output, source.data, size);
            return true;
        }
    }
    reader.failed = true;
    return false;
}

template <size_t Size>
void StoreGuestAddress(std::array<u32, NUM_USER_DATA_REGS>& user_data, u32 index,
                       const std::array<u32, Size>& source) {
    const VAddr address = reinterpret_cast<VAddr>(source.data());
    user_data[index] = static_cast<u32>(address);
    user_data[index + 1U] = static_cast<u32>(address >> 32);
}

Info MakeReservationInfo(std::span<const u32> user_data, u32 flattened_size,
                         std::initializer_list<PersistentSrtInfo::SrtRangeReservation>
                             reservations) {
    Info info{};
    info.user_data = user_data;
    info.srt_info.flattened_bufsize_dw = flattened_size;
    info.srt_info.range_reservations.assign(reservations);
    return info;
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
    const SrtMemoryReader reader{ReadWalkerRange, &recorder, &recorder.failed};

    ASSERT_TRUE(Optimization::FlattenExtendedUserdataPass(fixture.program, reader));

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
    const SrtMemoryReader reader{ReadWalkerRange, &recorder, &recorder.failed};

    ASSERT_TRUE(Optimization::FlattenExtendedUserdataPass(fixture.program, reader));
    ASSERT_EQ(recorder.ranges.size(), 2U);

    fixture.PointRootAt(fixture.stale_child);
    fixture.fresh_child[130] = 0U;
    fixture.fresh_child[131] = 0U;
    recorder.ranges.clear();
    ASSERT_TRUE(fixture.info.RefreshFlatBuf(reader));

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
    const SrtMemoryReader reader{ReadWalkerRange, &recorder, &recorder.failed};

    EXPECT_FALSE(Optimization::FlattenExtendedUserdataPass(fixture.program, reader));

    ASSERT_EQ(recorder.ranges.size(), 1U);
    EXPECT_EQ(recorder.ranges[0],
              std::make_pair(reinterpret_cast<VAddr>(fixture.root.data() + 2), 8ULL));
    ASSERT_GE(fixture.info.flattened_ud_buf.size(), 20U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[16], 0U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[17], 0U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[18], 0U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[19], 0U);
}

TEST(SrtWalker, RefusedLeafRangePublishesZerosInsteadOfStaleGuestData) {
    WalkerFixture fixture;
    fixture.PointRootAt(fixture.stale_child);
    fixture.stale_child[130] = 0xDEADBEEFU;
    fixture.stale_child[131] = 0xBAD0CAFEU;

    WalkerSyncRecorder recorder{.fixture = &fixture, .reject_child_data = true};
    const SrtMemoryReader reader{ReadWalkerRange, &recorder, &recorder.failed};

    EXPECT_FALSE(Optimization::FlattenExtendedUserdataPass(fixture.program, reader));

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

    bool failed{};
    const SrtMemoryReader reader{RejectWalkerRange, &failed, &failed};
    EXPECT_FALSE(Optimization::FlattenExtendedUserdataPass(program, reader));
    ASSERT_GE(info.flattened_ud_buf.size(), NUM_USER_DATA_REGS + 64U);
    for (u32 i = 0; i < 64; ++i) {
        EXPECT_EQ(info.flattened_ud_buf[NUM_USER_DATA_REGS + i], 0U);
    }
}

TEST(SrtWalker, RefreshCopiesEveryDeclaredReservationThroughOneReader) {
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    const std::array<u32, 6> source{0x100U, 0x101U, 0x102U, 0x103U, 0x104U, 0x105U};
    StoreGuestAddress(user_data, 0, source);
    auto info = MakeReservationInfo(
        user_data, 20, {{.pointer_lo_flat_index = 0,
                         .source_offset_dw = 2,
                         .destination_flat_offset_dw = 16,
                         .span_dw = 4}});
    ReservationReader capture{{ReservationSource{
        .address = reinterpret_cast<VAddr>(source.data() + 2),
        .data = source.data() + 2,
        .size = 4 * sizeof(u32),
    }}};
    const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};

    ASSERT_TRUE(info.RefreshFlatBuf(reader));
    ASSERT_EQ(capture.ranges.size(), 1U);
    EXPECT_EQ(capture.ranges[0],
              std::make_pair(reinterpret_cast<VAddr>(source.data() + 2), 16ULL));
    EXPECT_TRUE(std::ranges::equal(std::span(info.flattened_ud_buf).subspan(16, 4),
                                   std::span(source).subspan(2, 4)));
}

TEST(SrtWalker, MissingReaderClearsAReservedSnapshotAndRefusesProof) {
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    const std::array<u32, 2> source{0xAABBCCDDU, 0x11223344U};
    StoreGuestAddress(user_data, 0, source);
    auto info = MakeReservationInfo(
        user_data, 18, {{.pointer_lo_flat_index = 0,
                         .source_offset_dw = 0,
                         .destination_flat_offset_dw = 16,
                         .span_dw = 2}});
    info.flattened_ud_buf.assign(18, 0xDEADBEEFU);

    EXPECT_FALSE(info.RefreshFlatBuf());
    ASSERT_EQ(info.flattened_ud_buf.size(), 18U);
    EXPECT_EQ(info.flattened_ud_buf[16], 0U);
    EXPECT_EQ(info.flattened_ud_buf[17], 0U);
}

TEST(SrtWalker, AnyRefusedReservationClearsTheWholeTransactionalSnapshot) {
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    const std::array<u32, 2> first{0x11111111U, 0x22222222U};
    const std::array<u32, 2> second{0x33333333U, 0x44444444U};
    StoreGuestAddress(user_data, 0, first);
    StoreGuestAddress(user_data, 2, second);
    auto info = MakeReservationInfo(
        user_data, 20,
        {{.pointer_lo_flat_index = 0,
          .source_offset_dw = 0,
          .destination_flat_offset_dw = 16,
          .span_dw = 2},
         {.pointer_lo_flat_index = 2,
          .source_offset_dw = 0,
          .destination_flat_offset_dw = 18,
          .span_dw = 2}});
    ReservationReader capture{
        .sources = {{reinterpret_cast<VAddr>(first.data()), first.data(), sizeof(first)},
                    {reinterpret_cast<VAddr>(second.data()), second.data(), sizeof(second)}},
        .reject_call = 1,
    };
    const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};

    EXPECT_FALSE(info.RefreshFlatBuf(reader));
    ASSERT_EQ(capture.ranges.size(), 2U);
    EXPECT_TRUE(std::ranges::equal(std::span(info.flattened_ud_buf).subspan(16, 4),
                                   std::array<u32, 4>{0U, 0U, 0U, 0U}));
}

TEST(SrtWalker, InvalidReservationMetadataIsRejectedBeforeAllocationOrRead) {
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    auto info = MakeReservationInfo(
        user_data, 18, {{.pointer_lo_flat_index = 0,
                         .source_offset_dw = std::numeric_limits<u32>::max(),
                         .destination_flat_offset_dw = 16,
                         .span_dw = 2}});
    ReservationReader capture{};
    const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};

    EXPECT_FALSE(info.RefreshFlatBuf(reader));
    EXPECT_TRUE(info.flattened_ud_buf.empty());
    EXPECT_TRUE(capture.ranges.empty());
}

} // namespace
} // namespace Shader
