// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <cstring>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "common/arch.h"
#include "core/emulator_settings.h"

#if defined(_WIN32) && defined(ARCH_X86_64)
#include <windows.h>
#endif

#include "core/signals.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

namespace Shader {
namespace {

class DirectMemoryAccessSettingGuard {
public:
    explicit DirectMemoryAccessSettingGuard(bool enabled)
        : previous{EmulatorSettings.IsDirectMemoryAccessEnabled()} {
        EmulatorSettings.SetDirectMemoryAccessEnabled(enabled);
    }

    ~DirectMemoryAccessSettingGuard() {
        EmulatorSettings.SetDirectMemoryAccessEnabled(previous);
    }

private:
    bool previous;
};

#if defined(_WIN32) && defined(ARCH_X86_64)
TEST(SrtFaultRecovery, CachedWalkerRegistrationRecoversOnlyTheFaultingInvocation) {
    // mov r10d,[rdi]; mov [rsi],r10d; ret
    constexpr std::array<u8, 7> code{0x44, 0x8B, 0x17, 0x44, 0x89, 0x16, 0xC3};
    const auto walker = RegisterWalkerCode(code.data(), code.size());
    const auto* const walker_code = reinterpret_cast<const u8*>(walker);
    const std::array<u8, code.size()> before{
        walker_code[0], walker_code[1], walker_code[2], walker_code[3],
        walker_code[4], walker_code[5], walker_code[6],
    };

    CONTEXT registers{};
    registers.ContextFlags = CONTEXT_ALL;
    registers.Rip = reinterpret_cast<DWORD64>(walker_code);
    registers.R10 = 0x1122334455667788ULL;
    EXCEPTION_RECORD record{};
    record.ExceptionCode = EXCEPTION_ACCESS_VIOLATION;
    record.NumberParameters = 2;
    record.ExceptionInformation[0] = 0;
    record.ExceptionInformation[1] = 0xDEADBEEFULL;
    EXCEPTION_POINTERS exception{&record, &registers};

    EXPECT_TRUE(Core::Signals::Instance()->DispatchAccessViolation(
        &exception, reinterpret_cast<void*>(record.ExceptionInformation[1])));
    EXPECT_EQ(registers.R10, 0U);
    EXPECT_EQ(registers.Rip, reinterpret_cast<DWORD64>(walker_code + 3));
    EXPECT_TRUE(std::ranges::equal(before, std::span{walker_code, code.size()}));

    u32 source = 0xA1B2C3D4U;
    u32 destination = 0;
    walker(&source, &destination, nullptr, nullptr);
    EXPECT_EQ(destination, source);
}

TEST(SrtFaultRecovery, PointerLoadFaultZerosRdiAndAdvancesPastTheRead) {
    // mov rdi,[rdi]; ret
    constexpr std::array<u8, 4> code{0x48, 0x8B, 0x3F, 0xC3};
    const auto walker = RegisterWalkerCode(code.data(), code.size());
    const auto* const walker_code = reinterpret_cast<const u8*>(walker);

    CONTEXT registers{};
    registers.ContextFlags = CONTEXT_ALL;
    registers.Rip = reinterpret_cast<DWORD64>(walker_code);
    registers.Rdi = 0xDEADBEEFULL;
    EXCEPTION_RECORD record{};
    record.ExceptionCode = EXCEPTION_ACCESS_VIOLATION;
    record.NumberParameters = 2;
    record.ExceptionInformation[0] = 0;
    EXCEPTION_POINTERS exception{&record, &registers};

    EXPECT_TRUE(Core::Signals::Instance()->DispatchAccessViolation(&exception, nullptr));
    EXPECT_EQ(registers.Rdi, 0U);
    EXPECT_EQ(registers.Rip, reinterpret_cast<DWORD64>(walker_code + 3));
    EXPECT_TRUE(std::ranges::equal(code, std::span{walker_code, code.size()}));
}

TEST(SrtFaultRecovery, RejectsWriteFaultsAndNonGuestMemoryLoads) {
    // mov rdi,[rsp]; ret -- a stack restore is never a recoverable guest SRT read.
    constexpr std::array<u8, 5> code{0x48, 0x8B, 0x3C, 0x24, 0xC3};
    const auto walker = RegisterWalkerCode(code.data(), code.size());
    const auto* const walker_code = reinterpret_cast<const u8*>(walker);

    CONTEXT registers{};
    registers.ContextFlags = CONTEXT_ALL;
    registers.Rip = reinterpret_cast<DWORD64>(walker_code);
    registers.Rdi = 0x1122334455667788ULL;
    EXCEPTION_RECORD record{};
    record.ExceptionCode = EXCEPTION_ACCESS_VIOLATION;
    record.NumberParameters = 2;
    record.ExceptionInformation[0] = 0;
    EXCEPTION_POINTERS exception{&record, &registers};

    EXPECT_FALSE(Core::Signals::Instance()->DispatchAccessViolation(&exception, nullptr));
    EXPECT_EQ(registers.Rdi, 0x1122334455667788ULL);
    EXPECT_EQ(registers.Rip, reinterpret_cast<DWORD64>(walker_code));

    // Even a supported instruction must not claim a write access violation.
    constexpr std::array<u8, 4> read_code{0x48, 0x8B, 0x3F, 0xC3};
    const auto read_walker = RegisterWalkerCode(read_code.data(), read_code.size());
    registers.Rip = reinterpret_cast<DWORD64>(read_walker);
    record.ExceptionInformation[0] = 1;
    EXPECT_FALSE(Core::Signals::Instance()->DispatchAccessViolation(&exception, nullptr));
    EXPECT_EQ(registers.Rdi, 0x1122334455667788ULL);
    EXPECT_EQ(registers.Rip, reinterpret_cast<DWORD64>(read_walker));
}
#endif

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
        const auto root_pointer =
            ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0),
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

bool PS4_SYSV_ABI SynchronizeWalkerRange(void* opaque, VAddr address, u64 size) {
    auto& recorder = *static_cast<WalkerSyncRecorder*>(opaque);
    auto& fixture = *recorder.fixture;
    recorder.ranges.emplace_back(address, size);

    const VAddr root_pointer_address = reinterpret_cast<VAddr>(fixture.root.data() + 2);
    if (address == root_pointer_address && size == 8) {
        if (recorder.reject_root_pointer) {
            return false;
        }
        fixture.PointRootAt(fixture.fresh_child);
        return true;
    }

    const VAddr child_data_address = reinterpret_cast<VAddr>(fixture.fresh_child.data() + 130);
    if (address == child_data_address && size == 8) {
        if (recorder.reject_child_data) {
            return false;
        }
        fixture.fresh_child[130] = recorder.fresh_x;
        fixture.fresh_child[131] = recorder.fresh_y;
        return true;
    }
    return false;
}

bool PS4_SYSV_ABI RejectWalkerRange(void*, VAddr, u64) {
    return false;
}

TEST(SrtWalker, SynchronizesParentBeforeNestedEightBytePointerLoad) {
    WalkerFixture fixture;
    fixture.PointRootAt(fixture.stale_child);
    fixture.stale_child[130] = 250U;
    fixture.stale_child[131] = 0U;

    WalkerSyncRecorder recorder{
        .fixture = &fixture,
        .fresh_x = std::bit_cast<u32>(1.0F / 1920.0F),
        .fresh_y = std::bit_cast<u32>(1.0F / 1080.0F),
    };
    const SrtMemorySynchronizer synchronizer{
        .callback = SynchronizeWalkerRange,
        .context = &recorder,
    };

    Optimization::FlattenExtendedUserdataPass(fixture.program, synchronizer);

    ASSERT_EQ(recorder.ranges.size(), 2U);
    EXPECT_EQ(recorder.ranges[0],
              std::make_pair(reinterpret_cast<VAddr>(fixture.root.data() + 2), 8ULL));
    EXPECT_EQ(recorder.ranges[1],
              std::make_pair(reinterpret_cast<VAddr>(fixture.fresh_child.data() + 130), 8ULL));
    ASSERT_GE(fixture.info.flattened_ud_buf.size(), 20U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[18], recorder.fresh_x);
    EXPECT_EQ(fixture.info.flattened_ud_buf[19], recorder.fresh_y);
}

TEST(SrtWalker, UsesSameSynchronizerForInitialAndCachedRefresh) {
    WalkerFixture fixture;
    fixture.PointRootAt(fixture.stale_child);

    WalkerSyncRecorder recorder{
        .fixture = &fixture,
        .fresh_x = 0x3A112233U,
        .fresh_y = 0x3A445566U,
    };
    const SrtMemorySynchronizer synchronizer{
        .callback = SynchronizeWalkerRange,
        .context = &recorder,
    };

    Optimization::FlattenExtendedUserdataPass(fixture.program, synchronizer);
    ASSERT_EQ(recorder.ranges.size(), 2U);

    fixture.PointRootAt(fixture.stale_child);
    fixture.fresh_child[130] = 0U;
    fixture.fresh_child[131] = 0U;
    recorder.ranges.clear();
    fixture.info.RefreshFlatBuf(synchronizer);

    ASSERT_EQ(recorder.ranges.size(), 2U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[18], recorder.fresh_x);
    EXPECT_EQ(fixture.info.flattened_ud_buf[19], recorder.fresh_y);
}

TEST(SrtWalker, RefusedParentRangeDoesNotFollowOrPublishTheStaleNestedPointer) {
    WalkerFixture fixture;
    fixture.PointRootAt(fixture.stale_child);
    fixture.stale_child[130] = 0xDEADBEEFU;
    fixture.stale_child[131] = 0xBAD0CAFEU;

    WalkerSyncRecorder recorder{
        .fixture = &fixture,
        .reject_root_pointer = true,
    };
    const SrtMemorySynchronizer synchronizer{
        .callback = SynchronizeWalkerRange,
        .context = &recorder,
    };

    Optimization::FlattenExtendedUserdataPass(fixture.program, synchronizer);

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

    WalkerSyncRecorder recorder{
        .fixture = &fixture,
        .reject_child_data = true,
    };
    const SrtMemorySynchronizer synchronizer{
        .callback = SynchronizeWalkerRange,
        .context = &recorder,
    };

    Optimization::FlattenExtendedUserdataPass(fixture.program, synchronizer);

    ASSERT_EQ(recorder.ranges.size(), 2U);
    ASSERT_GE(fixture.info.flattened_ud_buf.size(), 20U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[18], 0U);
    EXPECT_EQ(fixture.info.flattened_ud_buf[19], 0U);
}

TEST(SrtWalker, LargeContiguousNodeUsesNearBranchesForRefusedRanges) {
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
    const auto root_pointer = ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0),
                                                    ir.GetUserData(IR::ScalarReg::S1));
    for (u32 i = 0; i < 64; ++i) {
        root[i] = 0xA0000000U + i;
        ir.Reference(ir.ReadConst(root_pointer, ir.Imm32(i)));
    }

    const SrtMemorySynchronizer synchronizer{
        .callback = RejectWalkerRange,
    };
    EXPECT_NO_THROW(Optimization::FlattenExtendedUserdataPass(program, synchronizer));
    ASSERT_GE(info.flattened_ud_buf.size(), NUM_USER_DATA_REGS + 64U);
    for (u32 i = 0; i < 64; ++i) {
        EXPECT_EQ(info.flattened_ud_buf[NUM_USER_DATA_REGS + i], 0U);
    }
}

TEST(ReadConstMemoryPolicy, DynamicReadConstRequiresPerShaderDmaWhenGlobalDmaIsDisabled) {
    DirectMemoryAccessSettingGuard direct_memory_access{false};
    Info info{};
    Pools pools{};
    IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);

    IR::IREmitter ir{*block};
    const auto pointer = ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0),
                                               ir.GetUserData(IR::ScalarReg::S1));
    ir.Reference(ir.ReadConst(pointer, ir.GetUserData(IR::ScalarReg::S2)));

    Optimization::CollectShaderInfoPass(program, Profile{});

    EXPECT_TRUE(info.uses_dma);
    EXPECT_TRUE(True(info.readconst_types & Info::ReadConstType::Dynamic));
}

TEST(ReadConstMemoryPolicy, FlattenedReadConstHonorsGlobalDmaDisable) {
    DirectMemoryAccessSettingGuard direct_memory_access{false};
    Info info{};
    Pools pools{};
    IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);

    IR::IREmitter ir{*block};
    const auto pointer = ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0),
                                               ir.GetUserData(IR::ScalarReg::S1));
    const auto flattened_read = ir.ReadConst(pointer, ir.Imm32(4U));
    flattened_read.InstRecursive()->SetFlags<u32>(NUM_USER_DATA_REGS);
    ir.Reference(flattened_read);

    Optimization::CollectShaderInfoPass(program, Profile{});

    EXPECT_FALSE(info.uses_dma);
    EXPECT_EQ(info.readconst_types, Info::ReadConstType::None);
}

TEST(ReadConstMemoryPolicy, Addr64BeforeFlattenedReadStillPublishesOneFlatBuffer) {
    DirectMemoryAccessSettingGuard direct_memory_access{false};
    Info info{};
    Pools pools{};
    IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);

    IR::IREmitter ir{*block};
    const auto handle = ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S4),
                                              ir.GetUserData(IR::ScalarReg::S5),
                                              ir.GetUserData(IR::ScalarReg::S6),
                                              ir.GetUserData(IR::ScalarReg::S7));
    const auto pointer = ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0),
                                               ir.GetUserData(IR::ScalarReg::S1));
    ir.Reference(ir.ReadConstBufferAddr64(handle, pointer, ir.Imm32(0U), {}));
    const auto flattened_read = ir.ReadConst(pointer, ir.Imm32(4U));
    flattened_read.InstRecursive()->SetFlags<u32>(NUM_USER_DATA_REGS);
    ir.Reference(flattened_read);

    Optimization::CollectShaderInfoPass(program, Profile{});

    EXPECT_EQ(std::ranges::count_if(info.buffers, [](const auto& buffer) {
                  return buffer.buffer_type == BufferType::Flatbuf;
              }),
              1U);
    EXPECT_FALSE(info.uses_dma);
}

} // namespace
} // namespace Shader
