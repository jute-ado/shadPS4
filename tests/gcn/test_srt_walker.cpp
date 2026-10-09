// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <cstring>
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
};

bool PS4_SYSV_ABI ReadWalkerRange(void* opaque, VAddr address, void* output, u64 size) {
    auto& recorder = *static_cast<WalkerSyncRecorder*>(opaque);
    auto& fixture = *recorder.fixture;
    recorder.ranges.emplace_back(address, size);

    const VAddr root_pointer_address = reinterpret_cast<VAddr>(fixture.root.data() + 2);
    if (address == root_pointer_address && size == 8) {
        if (recorder.reject_root_pointer) {
            return false;
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
    EXPECT_EQ(fixture.root[2], static_cast<u32>(reinterpret_cast<VAddr>(fixture.stale_child.data())));
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

    ASSERT_EQ(recorder.ranges.size(), 2U);
    EXPECT_EQ(recorder.ranges[0],
              std::make_pair(reinterpret_cast<VAddr>(fixture.root.data() + 2), 8ULL));
    EXPECT_EQ(recorder.ranges[1], std::make_pair(0x208ULL, 8ULL));
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

} // namespace
} // namespace Shader
