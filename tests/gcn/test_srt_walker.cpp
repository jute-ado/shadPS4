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

#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

namespace Shader {
namespace {

struct ScopedDirectMemoryAccess {
    const bool previous = EmulatorSettings.IsDirectMemoryAccessEnabled();

    explicit ScopedDirectMemoryAccess(bool enabled) {
        EmulatorSettings.SetDirectMemoryAccessEnabled(enabled);
    }
    ~ScopedDirectMemoryAccess() {
        EmulatorSettings.SetDirectMemoryAccessEnabled(previous);
    }
};

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

Info MakeReservationInfo(
    std::span<const u32> user_data, u32 flattened_size,
    std::initializer_list<PersistentSrtInfo::SrtRangeReservation> reservations) {
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

TEST(SrtWalker, PreservesConstantPointerBaseAdjustment) {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.user_data = user_data;

    constexpr VAddr RootAddress = 0x1234000000ULL;
    constexpr u32 BaseAdjustment = 0x40U;
    constexpr u32 DwordOffset = 2U;
    constexpr VAddr ExpectedAddress = RootAddress + BaseAdjustment + DwordOffset * sizeof(u32);
    constexpr VAddr UnadjustedAddress = RootAddress + DwordOffset * sizeof(u32);
    user_data[0] = static_cast<u32>(RootAddress);
    user_data[1] = static_cast<u32>(RootAddress >> 32);

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);
    IR::IREmitter ir{*block};
    // There is no low-dword carry in this case. The IR address is exactly root + 0x40 + 2*4.
    const auto pointer =
        ir.CompositeConstruct(ir.IAdd(ir.GetUserData(IR::ScalarReg::S0), ir.Imm32(BaseAdjustment)),
                              ir.GetUserData(IR::ScalarReg::S1));
    const auto value = ir.ReadConst(pointer, ir.Imm32(DwordOffset));
    ir.Reference(value);

    const u32 expected = 0xAABBCCDDU;
    const u32 unadjusted = 0x11223344U;
    // Both addresses are readable, so a wrong flattening cannot hide behind a refused read.
    ReservationReader capture{
        .sources = {{ExpectedAddress, &expected, sizeof(expected)},
                    {UnadjustedAddress, &unadjusted, sizeof(unadjusted)}},
    };
    const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};

    ASSERT_TRUE(Optimization::FlattenExtendedUserdataPass(program, reader));
    ASSERT_EQ(capture.ranges.size(), 1U);
    EXPECT_EQ(capture.ranges[0], std::make_pair(ExpectedAddress, u64{sizeof(u32)}));
    const auto destination = value.InstRecursive()->Flags<u32>();
    ASSERT_LT(destination, info.flattened_ud_buf.size());
    EXPECT_EQ(info.flattened_ud_buf[destination], expected);
}

TEST(SrtWalker, PointerAdjustmentUsesExactLowWrapAndExplicitHighCarry) {
    struct Case {
        u32 low;
        u32 high;
        u32 low_adjust;
        u32 high_adjust;
        bool carry;
        VAddr expected;
    };
    constexpr std::array cases{
        Case{0x34000000U, 0x12U, 0U, 0U, false, 0x1234000008ULL},
        Case{0xFFFFFFE0U, 0x12U, 0x40U, 0U, false, 0x1200000028ULL},
        Case{0xFFFFFFE0U, 0x12U, 0x40U, 0U, true, 0x1300000028ULL},
        Case{0xFFFFFFE0U, 0x12U, 0x40U, 2U, true, 0x1500000028ULL},
        Case{0xFFFFFFE0U, 0xFFFFFFFFU, 0x40U, 0U, true, 0x28ULL},
    };
    for (const auto& test : cases) {
        SCOPED_TRACE(test.expected);
        Info info{};
        Pools pools{};
        IR::Program program{info};
        std::array<u32, NUM_USER_DATA_REGS> user_data{};
        user_data[0] = test.low;
        user_data[1] = test.high;
        info.user_data = user_data;
        auto* const block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        program.post_order_blocks.push_back(block);
        IR::IREmitter ir{*block};
        const auto original_low = ir.GetUserData(IR::ScalarReg::S0);
        const IR::U32 low{ir.IAdd(original_low, ir.Imm32(test.low_adjust))};
        IR::U32 high{ir.IAdd(ir.GetUserData(IR::ScalarReg::S1), ir.Imm32(test.high_adjust))};
        if (test.carry) {
            const IR::U32 carry{
                ir.Select(ir.ILessThan(low, original_low, false), ir.Imm32(1U), ir.Imm32(0U))};
            high = IR::U32{ir.IAdd(high, carry)};
        }
        const auto value = ir.ReadConst(ir.CompositeConstruct(low, high), ir.Imm32(2U));
        ir.Reference(value);
        const u32 expected = 0xABCD1234U;
        ReservationReader capture{.sources = {{test.expected, &expected, sizeof(expected)}}};
        const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};

        ASSERT_TRUE(Optimization::FlattenExtendedUserdataPass(program, reader));
        ASSERT_EQ(capture.ranges.size(), 1U);
        EXPECT_EQ(capture.ranges[0].first, test.expected);
        EXPECT_EQ(info.flattened_ud_buf[value.InstRecursive()->Flags<u32>()], expected);
        capture.ranges.clear();
        ASSERT_TRUE(info.RefreshFlatBuf(reader));
        ASSERT_EQ(capture.ranges.size(), 1U);
        EXPECT_EQ(capture.ranges[0].first, test.expected);
    }
}

TEST(SrtWalker, PointerUsesItsActualNonAdjacentUserDataHighHalf) {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    user_data[0] = 0x34000000U;
    user_data[1] = 0x77U;
    user_data[7] = 0x12U;
    info.user_data = user_data;
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);
    IR::IREmitter ir{*block};
    const auto pointer =
        ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0), ir.GetUserData(IR::ScalarReg::S7));
    const auto value = ir.ReadConst(pointer, ir.Imm32(2U));
    ir.Reference(value);
    const u32 expected = 0xABCD1234U;
    ReservationReader capture{.sources = {{0x1234000008ULL, &expected, sizeof(expected)}}};
    const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};

    ASSERT_TRUE(Optimization::FlattenExtendedUserdataPass(program, reader));
    ASSERT_EQ(capture.ranges.size(), 1U);
    EXPECT_EQ(capture.ranges[0].first, 0x1234000008ULL);
    EXPECT_EQ(info.flattened_ud_buf[value.InstRecursive()->Flags<u32>()], expected);
}

TEST(SrtWalker, PointerSubtractionAndMasksPreserveWrappingScalarSemantics) {
    for (const auto [initial_low, address] :
         {std::pair{0x3400004FU, 0x1234000008ULL}, std::pair{0x20U, 0x12FFFFFFE8ULL}}) {
        SCOPED_TRACE(initial_low);
        Info info{};
        Pools pools{};
        IR::Program program{info};
        std::array<u32, NUM_USER_DATA_REGS> user_data{};
        user_data[0] = initial_low;
        user_data[1] = 0x00123412U;
        info.user_data = user_data;
        auto* const block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        program.post_order_blocks.push_back(block);
        IR::IREmitter ir{*block};
        const auto low = ir.BitwiseAnd(ir.ISub(ir.GetUserData(IR::ScalarReg::S0), ir.Imm32(0x40U)),
                                       ir.Imm32(0xFFFFFFF0U));
        const auto high = ir.BitwiseAnd(ir.GetUserData(IR::ScalarReg::S1), ir.Imm32(0xFFU));
        const auto value = ir.ReadConst(ir.CompositeConstruct(low, high), ir.Imm32(2U));
        ir.Reference(value);
        const u32 expected = 0xABCD1234U;
        ReservationReader capture{.sources = {{address, &expected, sizeof(expected)}}};
        const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};

        ASSERT_TRUE(Optimization::FlattenExtendedUserdataPass(program, reader));
        ASSERT_EQ(capture.ranges.size(), 1U);
        EXPECT_EQ(capture.ranges.front().first, address);
        EXPECT_EQ(info.flattened_ud_buf[value.InstRecursive()->Flags<u32>()], expected);
    }
}

TEST(SrtWalker, UniqueSourcePhiWithSelfEdgePreservesTheActualPointer) {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    user_data[0] = 0x34000000U;
    user_data[1] = 0x12U;
    info.user_data = user_data;
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);
    IR::IREmitter ir{*block};
    const auto source = ir.GetUserData(IR::ScalarReg::S0);
    auto* const phi = &*block->PrependNewInst(block->begin(), IR::Opcode::Phi);
    phi->SetFlags(IR::Type::U32);
    phi->AddPhiOperand(block, source);
    phi->AddPhiOperand(block, IR::Value{phi});
    const auto value = ir.ReadConst(
        ir.CompositeConstruct(IR::U32{phi}, ir.GetUserData(IR::ScalarReg::S1)), ir.Imm32(2U));
    ir.Reference(value);
    const u32 expected = 0xABCD1234U;
    ReservationReader capture{.sources = {{0x1234000008ULL, &expected, sizeof(expected)}}};
    const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};

    ASSERT_TRUE(Optimization::FlattenExtendedUserdataPass(program, reader));
    ASSERT_EQ(capture.ranges.size(), 1U);
    EXPECT_EQ(capture.ranges.front().first, 0x1234000008ULL);
    EXPECT_EQ(info.flattened_ud_buf[value.InstRecursive()->Flags<u32>()], expected);
}

TEST(SrtWalker, CachedRefreshReevaluatesSharedCarryWhenTheConditionChanges) {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    user_data[1] = 0x12U;
    info.user_data = user_data;
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);
    IR::IREmitter ir{*block};
    const auto original_low = ir.GetUserData(IR::ScalarReg::S0);
    const IR::U32 low{ir.IAdd(original_low, ir.Imm32(0x40U))};
    const auto condition = ir.ILessThan(low, original_low, false);
    const auto carry = ir.Select(condition, ir.Imm32(1U), ir.Imm32(0U));
    const auto high = ir.IAdd(ir.GetUserData(IR::ScalarReg::S1), IR::U32{carry});
    const auto value = ir.ReadConst(ir.CompositeConstruct(low, high), ir.Imm32(2U));
    ir.Reference(value);
    const u32 carried = 0x11112222U;
    const u32 uncarried = 0x33334444U;
    ReservationReader capture{.sources = {{0x1300000028ULL, &carried, sizeof(carried)},
                                          {0x1200001048ULL, &uncarried, sizeof(uncarried)}}};
    const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};
    user_data[0] = 0xFFFFFFE0U;

    ASSERT_TRUE(Optimization::FlattenExtendedUserdataPass(program, reader));
    const auto destination = value.InstRecursive()->Flags<u32>();
    EXPECT_EQ(info.flattened_ud_buf[destination], carried);

    user_data[0] = 0x1000U;
    capture.ranges.clear();
    ASSERT_TRUE(info.RefreshFlatBuf(reader));
    ASSERT_EQ(capture.ranges.size(), 1U);
    EXPECT_EQ(capture.ranges.front().first, 0x1200001048ULL);
    EXPECT_EQ(info.flattened_ud_buf[destination], uncarried);

    user_data[0] = 0xFFFFFFE0U;
    capture.ranges.clear();
    ASSERT_TRUE(info.RefreshFlatBuf(reader));
    ASSERT_EQ(capture.ranges.size(), 1U);
    EXPECT_EQ(capture.ranges.front().first, 0x1300000028ULL);
    EXPECT_EQ(info.flattened_ud_buf[destination], carried);
}

TEST(SrtWalker, DistinctPointerAdjustmentsDoNotAliasOneSourceOffset) {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    user_data[0] = 0x34000000U;
    user_data[1] = 0x12U;
    info.user_data = user_data;
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);
    IR::IREmitter ir{*block};
    const auto make_read = [&](u32 adjustment) {
        const auto pointer =
            ir.CompositeConstruct(ir.IAdd(ir.GetUserData(IR::ScalarReg::S0), ir.Imm32(adjustment)),
                                  ir.GetUserData(IR::ScalarReg::S1));
        const auto read = ir.ReadConst(pointer, ir.Imm32(2U));
        ir.Reference(read);
        return read;
    };
    const auto first = make_read(0x40U);
    const auto second = make_read(0x80U);
    const u32 first_value = 0x11112222U;
    const u32 second_value = 0x33334444U;
    ReservationReader capture{
        .sources = {{0x1234000048ULL, &first_value, sizeof(first_value)},
                    {0x1234000088ULL, &second_value, sizeof(second_value)}},
    };
    const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};

    ASSERT_TRUE(Optimization::FlattenExtendedUserdataPass(program, reader));
    ASSERT_EQ(capture.ranges.size(), 2U);
    EXPECT_EQ(info.flattened_ud_buf[first.InstRecursive()->Flags<u32>()], first_value);
    EXPECT_EQ(info.flattened_ud_buf[second.InstRecursive()->Flags<u32>()], second_value);
}

TEST(SrtWalker, NestedAdjustedPointerUsesNonAdjacentCapturedHalvesOnRefresh) {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    user_data[0] = 0x34000000U;
    user_data[1] = 0x12U;
    info.user_data = user_data;
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);
    IR::IREmitter ir{*block};
    const auto root =
        ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0), ir.GetUserData(IR::ScalarReg::S1));
    const auto child_low = ir.ReadConst(root, ir.Imm32(2U));
    // This unrelated captured dword sits between the flattened pointer halves too.
    ir.Reference(ir.ReadConst(root, ir.Imm32(3U)));
    const auto child_high = ir.ReadConst(root, ir.Imm32(7U));
    const auto child = ir.CompositeConstruct(ir.IAdd(child_low, ir.Imm32(0x40U)), child_high);
    const auto value = ir.ReadConst(child, ir.Imm32(2U));
    ir.Reference(value);

    std::array<u32, 2> low_and_unrelated{0x56000000U, 0x77U};
    const u32 high = 0x23U;
    const u32 expected_first = 0x11112222U;
    const u32 expected_second = 0x33334444U;
    ReservationReader capture{
        .sources = {{0x1234000008ULL, low_and_unrelated.data(), sizeof(low_and_unrelated)},
                    {0x123400001CULL, &high, sizeof(high)},
                    {0x2356000048ULL, &expected_first, sizeof(expected_first)},
                    {0x2367000048ULL, &expected_second, sizeof(expected_second)}},
    };
    const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};
    ASSERT_TRUE(Optimization::FlattenExtendedUserdataPass(program, reader));
    ASSERT_EQ(capture.ranges.size(), 3U);
    EXPECT_EQ(capture.ranges.back().first, 0x2356000048ULL);
    const auto destination = value.InstRecursive()->Flags<u32>();
    EXPECT_EQ(info.flattened_ud_buf[destination], expected_first);

    low_and_unrelated[0] = 0x67000000U;
    capture.ranges.clear();
    ASSERT_TRUE(info.RefreshFlatBuf(reader));
    ASSERT_EQ(capture.ranges.size(), 3U);
    EXPECT_EQ(capture.ranges.back().first, 0x2367000048ULL);
    EXPECT_EQ(info.flattened_ud_buf[destination], expected_second);
}

TEST(SrtWalker, UnsupportedPointerExpressionIsRefusedBeforeReadingAnAncestorAddress) {
    const ScopedDirectMemoryAccess dma{false};
    Info info{};
    Pools pools{};
    IR::Program program{info};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    user_data[0] = 0x34000000U;
    user_data[1] = 0x12U;
    info.user_data = user_data;
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);
    IR::IREmitter ir{*block};
    const auto pointer =
        ir.CompositeConstruct(ir.IMul(ir.GetUserData(IR::ScalarReg::S0), ir.Imm32(2U)),
                              ir.GetUserData(IR::ScalarReg::S1));
    ir.Reference(ir.ReadConst(pointer, ir.Imm32(2U)));
    const u32 ancestor_value = 0x11223344U;
    ReservationReader capture{
        .sources = {{0x1234000008ULL, &ancestor_value, sizeof(ancestor_value)}},
    };
    const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};

    EXPECT_FALSE(Optimization::FlattenExtendedUserdataPass(program, reader));
    EXPECT_TRUE(info.translation_failed);
    EXPECT_TRUE(capture.ranges.empty());
    EXPECT_EQ(info.srt_info.walker_func, nullptr);
}

TEST(SrtWalker, AmbiguousPointerPhiIsRefusedBeforePickingOneSource) {
    const ScopedDirectMemoryAccess dma{false};
    Info info{};
    Pools pools{};
    IR::Program program{info};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    info.user_data = user_data;
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);
    IR::IREmitter ir{*block};
    const auto first = ir.GetUserData(IR::ScalarReg::S0);
    const auto second = ir.GetUserData(IR::ScalarReg::S2);
    auto* const phi = &*block->PrependNewInst(block->begin(), IR::Opcode::Phi);
    phi->SetFlags(IR::Type::U32);
    phi->AddPhiOperand(block, first);
    phi->AddPhiOperand(block, second);
    const auto pointer = ir.CompositeConstruct(IR::U32{phi}, ir.GetUserData(IR::ScalarReg::S1));
    ir.Reference(ir.ReadConst(pointer, ir.Imm32(0U)));
    ReservationReader capture{};
    const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};

    EXPECT_FALSE(Optimization::FlattenExtendedUserdataPass(program, reader));
    EXPECT_TRUE(info.translation_failed);
    EXPECT_TRUE(capture.ranges.empty());
    EXPECT_EQ(info.srt_info.walker_func, nullptr);
}

TEST(SrtWalker, ResourceHandleOperandsExcludeShaderRuntimeData) {
    for (const auto [opcode, mask] : {std::pair{IR::Opcode::LoadBufferU32, 1U},
                                      {IR::Opcode::StoreBufferU8, 1U},
                                      {IR::Opcode::BufferAtomicIAdd32, 1U},
                                      {IR::Opcode::ReadConstBufferAddr64, 1U},
                                      {IR::Opcode::ImageRead, 1U},
                                      {IR::Opcode::ImageWrite, 1U},
                                      {IR::Opcode::ImageQueryLod, 1U},
                                      {IR::Opcode::ImageAtomicIAdd32, 1U},
                                      {IR::Opcode::ImageSampleRaw, 3U},
                                      {IR::Opcode::IAdd32, 0U},
                                      {IR::Opcode::ReadConst, 0U},
                                      {IR::Opcode::LoadSharedU32, 0U}}) {
        SCOPED_TRACE(static_cast<u32>(opcode));
        const IR::Inst inst{opcode, 0U};
        EXPECT_EQ(Optimization::GetResourceHandleOperandMask(inst), mask);
    }
}

TEST(SrtWalker, GpuComputedScalarDataReadsStayDynamicOnlyWhenNoHostDescriptorNeedsThem) {
    enum class Consumer {
        Data,
        NoDma,
        Buffer,
        Image,
        Sampler,
        ArithmeticBuffer,
        NestedBuffer,
        ImageCoordinates
    };
    for (const auto consumer :
         {Consumer::Data, Consumer::NoDma, Consumer::Buffer, Consumer::Image, Consumer::Sampler,
          Consumer::ArithmeticBuffer, Consumer::NestedBuffer, Consumer::ImageCoordinates}) {
        SCOPED_TRACE(static_cast<u32>(consumer));
        const ScopedDirectMemoryAccess dma{consumer != Consumer::NoDma};
        Info info{};
        Pools pools{};
        IR::Program program{info};
        std::array<u32, NUM_USER_DATA_REGS> user_data{};
        user_data[0] = 0x34000000U;
        user_data[1] = 0x12U;
        info.user_data = user_data;
        auto* const block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        program.post_order_blocks.push_back(block);
        IR::IREmitter ir{*block};
        const auto root = ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0),
                                                ir.GetUserData(IR::ScalarReg::S1));
        const auto low = ir.ReadConst(root, ir.Imm32(2U));
        const auto high = ir.ReadConst(root, ir.Imm32(3U));
        const IR::U32 gpu_value{ir.LoadShared(32, false, ir.Imm32(0U), false)};
        const IR::U32 runtime_low{ir.IAdd(low, ir.IMul(gpu_value, ir.Imm32(112U)))};
        const auto pointer = ir.CompositeConstruct(ir.ReadFirstLane(runtime_low), high);
        const auto read = ir.ReadConst(pointer, ir.Imm32(4U));
        IR::U32 handle_value = read;
        if (consumer == Consumer::ArithmeticBuffer) {
            handle_value = IR::U32{ir.IAdd(read, ir.Imm32(0x40U))};
        } else if (consumer == Consumer::NestedBuffer) {
            handle_value = ir.ReadConst(ir.CompositeConstruct(read, high), ir.Imm32(0U));
        }
        if (consumer == Consumer::Buffer || consumer == Consumer::ArithmeticBuffer ||
            consumer == Consumer::NestedBuffer) {
            const auto handle =
                ir.CompositeConstruct(handle_value, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U));
            const auto address = ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U));
            ir.Reference(ir.LoadBufferU32(1, handle, address, {}));
        } else if (consumer == Consumer::Image) {
            const auto coords = ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U));
            ir.Reference(ir.ImageRead(read, coords, ir.Imm32(0U), ir.Imm32(0U), {}));
        } else if (consumer == Consumer::ImageCoordinates) {
            const auto coords = ir.CompositeConstruct(read, ir.Imm32(0U));
            ir.Reference(ir.ImageRead(ir.GetUserData(IR::ScalarReg::S2), coords, ir.Imm32(0U),
                                      ir.Imm32(0U), {}));
        } else if (consumer == Consumer::Sampler) {
            const auto sampler =
                ir.CompositeConstruct(read, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U));
            ir.Reference(ir.ImageSampleRaw(ir.GetUserData(IR::ScalarReg::S2), sampler, ir.Imm32(0U),
                                           ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U), {}));
        } else {
            ir.Reference(read);
        }
        const std::array<u32, 2> captured_pointer{0x56000000U, 0x23U};
        ReservationReader capture{
            .sources = {{0x1234000008ULL, captured_pointer.data(), sizeof(captured_pointer)}}};
        const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};

        if (consumer == Consumer::Data || consumer == Consumer::ImageCoordinates) {
            ASSERT_TRUE(Optimization::FlattenExtendedUserdataPass(program, reader));
            EXPECT_FALSE(info.translation_failed);
            EXPECT_EQ(read.InstRecursive()->Flags<u32>(), 0U);
            EXPECT_EQ(read.InstRecursive()->Arg(0), pointer);
            ASSERT_EQ(capture.ranges.size(), 1U);
            EXPECT_EQ(capture.ranges.front(), std::make_pair(0x1234000008ULL, 8ULL));
            EXPECT_EQ(info.flattened_ud_buf[low.InstRecursive()->Flags<u32>()],
                      captured_pointer[0]);
            capture.ranges.clear();
            ASSERT_TRUE(info.RefreshFlatBuf(reader));
            EXPECT_EQ(capture.ranges.size(), 1U);
        } else {
            EXPECT_FALSE(Optimization::FlattenExtendedUserdataPass(program, reader));
            EXPECT_TRUE(info.translation_failed);
            EXPECT_TRUE(capture.ranges.empty());
            EXPECT_EQ(info.srt_info.walker_func, nullptr);
        }
    }
}

TEST(SrtWalker, SharedPointerArithmeticHasLinearGeneratedCodeSize) {
    struct Case {
        u32 low;
        u32 depth;
        VAddr address;
    };
    for (const auto [initial_low, depth, address] :
         {Case{0x1000U, 12U, 0x1201000008ULL}, Case{1U, 30U, 0x1240000008ULL}}) {
        SCOPED_TRACE(depth);
        Info info{};
        Pools pools{};
        IR::Program program{info};
        std::array<u32, NUM_USER_DATA_REGS> user_data{};
        user_data[0] = initial_low;
        user_data[1] = 0x12U;
        info.user_data = user_data;
        auto* const block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        program.post_order_blocks.push_back(block);
        IR::IREmitter ir{*block};
        auto low = ir.GetUserData(IR::ScalarReg::S0);
        for (u32 i = 0; i < depth; ++i) {
            low = IR::U32{ir.IAdd(low, low)};
        }
        const auto value = ir.ReadConst(
            ir.CompositeConstruct(low, ir.GetUserData(IR::ScalarReg::S1)), ir.Imm32(2U));
        ir.Reference(value);
        const u32 expected = 0xABCD1234U;
        ReservationReader capture{.sources = {{address, &expected, sizeof(expected)}}};
        const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};

        ASSERT_TRUE(Optimization::FlattenExtendedUserdataPass(program, reader));
        EXPECT_EQ(info.flattened_ud_buf[value.InstRecursive()->Flags<u32>()], expected);
        ASSERT_EQ(capture.ranges.size(), 1U);
        EXPECT_EQ(capture.ranges.front().first, address);
        // Thirty distinct additions must not expand into a billion duplicated subexpressions.
        EXPECT_LT(info.srt_info.walker_func_size, 4096U);
    }
}

TEST(SrtWalker, GpuComputedImageTableIndexRemainsRuntimeWhileBaseAndSamplerAreCaptured) {
    const ScopedDirectMemoryAccess dma{true};
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    Pools pools{};
    IR::Program program{info};
    Gcn::Block cfg_block{};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    user_data[0] = 0x34000000U;
    user_data[1] = 0x12U;
    info.user_data = user_data;
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    block->cfg_block = &cfg_block;
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);
    IR::IREmitter ir{*block};
    const auto root =
        ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0), ir.GetUserData(IR::ScalarReg::S1));
    const auto low = ir.ReadConst(root, ir.Imm32(2U));
    const auto high = ir.ReadConst(root, ir.Imm32(3U));
    const auto table = ir.CompositeConstruct(low, high);
    const IR::U32 gpu_value{ir.LoadShared(32, false, ir.Imm32(0U), false)};
    const IR::U32 runtime_low{ir.IAdd(low, ir.IMul(gpu_value, ir.Imm32(112U)))};
    const auto runtime_pointer = ir.CompositeConstruct(ir.ReadFirstLane(runtime_low), high);
    const auto runtime_read = ir.ReadConst(runtime_pointer, ir.Imm32(0U));
    const IR::U32 index{ir.BitwiseAnd(runtime_read, ir.Imm32(3U))};
    const auto byte_offset = ir.IAdd(ir.IMul(index, ir.Imm32(32U)), ir.Imm32(14480U));
    const auto handle = ir.ReadConst(table, ir.ShiftRightLogical(byte_offset, ir.Imm32(2U)));
    const auto sampler =
        ir.CompositeConstruct(ir.ReadConst(root, ir.Imm32(4U)), ir.ReadConst(root, ir.Imm32(5U)),
                              ir.ReadConst(root, ir.Imm32(6U)), ir.ReadConst(root, ir.Imm32(7U)));
    const auto coords =
        ir.CompositeConstruct(ir.Imm32(0.25f), ir.Imm32(0.5f), ir.Imm32(0.0f), ir.Imm32(0.0f));
    IR::TextureInstInfo texture_info{};
    texture_info.has_lod.Assign(true);
    ir.Reference(
        ir.ImageSampleRaw(handle, sampler, coords, coords, coords, ir.Imm32(0.0f), texture_info));
    const std::array<u32, 6> captured_root{0x56000000U, 0x23U, 0U, 0U, 0U, 0U};
    std::array<AmdGpu::Image, 4> images;
    for (u32 i = 0; i < images.size(); ++i) {
        images[i] = AmdGpu::Image::Null(false);
        images[i].base_address = (0x10000U + 0x10000U * i) >> 8U;
        images[i].width = 63U;
        images[i].height = 63U;
    }
    ReservationReader capture{
        .sources = {{0x1234000008ULL, captured_root.data(), sizeof(captured_root)},
                    {0x2356003890ULL, images.data(), sizeof(images)}}};
    const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};

    ASSERT_TRUE(Optimization::FlattenExtendedUserdataPass(program, reader));
    EXPECT_FALSE(info.translation_failed);
    EXPECT_EQ(runtime_read.InstRecursive()->Flags<u32>(), 0U);
    EXPECT_EQ(runtime_read.InstRecursive()->Arg(0), runtime_pointer);
    ASSERT_EQ(capture.ranges.size(), 2U);
    EXPECT_EQ(low.InstRecursive()->Flags<u32>(), 16U);
    EXPECT_EQ(high.InstRecursive()->Flags<u32>(), 17U);
    Profile profile{};
    profile.max_sampled_images_per_stage = 256U;
    profile.max_sampled_images_per_set = 256U;
    profile.max_resources_per_stage = 256U;
    profile.max_sampled_images_per_descriptor_heap_set = 256U;
    profile.supports_sampled_image_array_nonuniform_indexing = true;
    Optimization::ResourceTrackingPass(program, profile, true);

    ASSERT_FALSE(info.translation_failed);
    ASSERT_EQ(info.images.size(), 1U);
    EXPECT_EQ(info.images.front().dynamic_descriptor_count, 4U);
    EXPECT_EQ(info.images.front().sharp_idx, 22U);
    ASSERT_EQ(info.samplers.size(), 1U);
    EXPECT_EQ(info.samplers.front().sharp_idx, 18U);
    const auto dynamic_sample = std::ranges::find_if(*block, [](const IR::Inst& inst) {
        return inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLodDynamic;
    });
    ASSERT_NE(dynamic_sample, block->end());
    EXPECT_EQ(dynamic_sample->Arg(1), index);
}

TEST(SrtWalker, PointerFormImageTableDoesNotCaptureItsGpuComputedRecordData) {
    for (const bool direct_memory : {true, false}) {
        const ScopedDirectMemoryAccess dma{direct_memory};
        Info info{};
        info.stage = Stage::Compute;
        info.l_stage = LogicalStage::Compute;
        Pools pools{};
        IR::Program program{info};
        Gcn::Block cfg_block{};
        constexpr u64 RootAddress = 0x12FFFFFFF0ULL;
        std::array<u32, NUM_USER_DATA_REGS> user_data{};
        user_data[0] = static_cast<u32>(RootAddress);
        user_data[1] = static_cast<u32>(RootAddress >> 32U);
        info.user_data = user_data;
        auto* const block = pools.block_pool.Create(pools.inst_pool);
        block->cfg_block = &cfg_block;
        program.blocks.push_back(block);
        program.post_order_blocks.push_back(block);
        IR::IREmitter ir{*block};
        const auto base_low = ir.GetUserData(IR::ScalarReg::S0);
        const auto base_high = ir.GetUserData(IR::ScalarReg::S1);
        const auto root = ir.CompositeConstruct(base_low, base_high);
        const auto add_to_root = [&](IR::U32 byte_offset) {
            const IR::U32 low{ir.IAdd(byte_offset, base_low)};
            const IR::U32 carry{
                ir.Select(ir.ILessThan(low, byte_offset, false), ir.Imm32(1U), ir.Imm32(0U))};
            return ir.CompositeConstruct(low, ir.IAdd(base_high, carry));
        };
        const auto count = ir.ReadConst(root, ir.Imm32(2U));
        const IR::U32 gpu_value{ir.LoadShared(32, false, ir.Imm32(0U), false)};
        const auto record_index = ir.ISub(count, ir.ReadFirstLane(gpu_value));
        const auto record_pointer = add_to_root(IR::U32{ir.IMul(record_index, ir.Imm32(224U))});
        const auto record_data = ir.ReadConst(record_pointer, ir.Imm32(36U));
        // This dependency models data feeding a later table selection (also possible through a
        // loop phi). The record itself is not a host descriptor and must remain a runtime read.
        const IR::U32 index{ir.BitwiseAnd(record_data, ir.Imm32(3U))};
        const auto image_pointer =
            add_to_root(IR::U32{ir.IAdd(ir.IMul(index, ir.Imm32(32U)), ir.Imm32(0x100U))});
        const auto handle = ir.ReadConst(image_pointer, ir.Imm32(0U));
        const auto sampler = ir.CompositeConstruct(
            ir.ReadConst(root, ir.Imm32(4U)), ir.ReadConst(root, ir.Imm32(5U)),
            ir.ReadConst(root, ir.Imm32(6U)), ir.ReadConst(root, ir.Imm32(7U)));
        const auto coords =
            ir.CompositeConstruct(ir.Imm32(0.25f), ir.Imm32(0.5f), ir.Imm32(0.0f), ir.Imm32(0.0f));
        IR::TextureInstInfo texture_info{};
        texture_info.has_lod.Assign(true);
        ir.Reference(ir.ImageSampleRaw(handle, sampler, coords, coords, coords, ir.Imm32(0.0f),
                                       texture_info));
        const std::array<u32, 6> captured_root{7U, 0U, 0U, 0U, 0U, 0U};
        std::array<AmdGpu::Image, 4> images;
        for (u32 i = 0; i < images.size(); ++i) {
            images[i] = AmdGpu::Image::Null(false);
            images[i].base_address = (0x10000U + 0x10000U * i) >> 8U;
            images[i].width = 63U;
            images[i].height = 63U;
        }
        ReservationReader capture{
            .sources = {{RootAddress + 8U, captured_root.data(), sizeof(u32)},
                        {RootAddress + 16U, captured_root.data() + 2U, 4U * sizeof(u32)},
                        {RootAddress + 0x100U, images.data(), sizeof(images)}}};
        const SrtMemoryReader reader{ReadReservationRange, &capture, &capture.failed};

        if (!direct_memory) {
            EXPECT_FALSE(Optimization::FlattenExtendedUserdataPass(program, reader));
            EXPECT_TRUE(info.translation_failed);
            EXPECT_TRUE(capture.ranges.empty());
            continue;
        }
        ASSERT_TRUE(Optimization::FlattenExtendedUserdataPass(program, reader));
        ASSERT_FALSE(info.translation_failed);
        EXPECT_EQ(record_data.InstRecursive()->Flags<u32>(), 0U);
        EXPECT_EQ(record_data.InstRecursive()->Arg(0), record_pointer);
        EXPECT_EQ(handle.InstRecursive()->Flags<u32>(), 0U);
        ASSERT_EQ(capture.ranges.size(), 3U);
        EXPECT_EQ(capture.ranges.back().first, RootAddress + 0x100U);
        Profile profile{};
        profile.max_sampled_images_per_stage = 256U;
        profile.max_sampled_images_per_set = 256U;
        profile.max_resources_per_stage = 256U;
        profile.max_sampled_images_per_descriptor_heap_set = 256U;
        profile.supports_sampled_image_array_nonuniform_indexing = true;
        Optimization::ResourceTrackingPass(program, profile, true);

        ASSERT_FALSE(info.translation_failed);
        ASSERT_EQ(info.images.size(), 1U);
        EXPECT_EQ(info.images.front().dynamic_descriptor_count, 4U);
        ASSERT_EQ(info.samplers.size(), 1U);
        const auto dynamic_sample = std::ranges::find_if(*block, [](const IR::Inst& inst) {
            return inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLodDynamic;
        });
        ASSERT_NE(dynamic_sample, block->end());
        EXPECT_EQ(dynamic_sample->Arg(1), index);
        // Repeated snapshot refresh must preserve the same bulk source and runtime data identity.
        capture.ranges.clear();
        ASSERT_TRUE(info.RefreshFlatBuf(reader));
        ASSERT_EQ(capture.ranges.size(), 3U);
        EXPECT_EQ(capture.ranges.back().first, RootAddress + 0x100U);
        EXPECT_EQ(record_data.InstRecursive()->Arg(0), record_pointer);
    }
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
    auto info = MakeReservationInfo(user_data, 20,
                                    {{.pointer_lo_flat_index = 0,
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
    EXPECT_EQ(capture.ranges[0], std::make_pair(reinterpret_cast<VAddr>(source.data() + 2), 16ULL));
    EXPECT_TRUE(std::ranges::equal(std::span(info.flattened_ud_buf).subspan(16, 4),
                                   std::span(source).subspan(2, 4)));
}

TEST(SrtWalker, MissingReaderClearsAReservedSnapshotAndRefusesProof) {
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    const std::array<u32, 2> source{0xAABBCCDDU, 0x11223344U};
    StoreGuestAddress(user_data, 0, source);
    auto info = MakeReservationInfo(user_data, 18,
                                    {{.pointer_lo_flat_index = 0,
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
    auto info = MakeReservationInfo(user_data, 20,
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
    auto info = MakeReservationInfo(user_data, 18,
                                    {{.pointer_lo_flat_index = 0,
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
