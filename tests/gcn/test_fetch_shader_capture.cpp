// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>

#include <gtest/gtest.h>

#include "instructions.hpp"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Gcn {
namespace {

struct GuestProgram {
    VAddr address{};
    std::span<const u32> words;
    u32 calls{};
    std::array<VAddr, 4> read_addresses{};
    std::array<u64, 4> read_sizes{};
};

struct RepeatingProgram {
    VAddr address{};
    u32 word{};
    u32 calls{};
};

bool PS4_SYSV_ABI ReadGuestProgram(void* context, VAddr address, void* output, u64 size) {
    auto& program = *static_cast<GuestProgram*>(context);
    const u32 call = program.calls++;
    if (call < program.read_addresses.size()) {
        program.read_addresses[call] = address;
        program.read_sizes[call] = size;
    }
    const auto bytes = std::as_bytes(program.words);
    if (address < program.address || size > bytes.size() ||
        address - program.address > bytes.size() - size) {
        return false;
    }
    std::memcpy(output, bytes.data() + (address - program.address), size);
    return true;
}

bool PS4_SYSV_ABI RefuseGuestProgram(void* context, VAddr, void*, u64) {
    ++*static_cast<u32*>(context);
    return false;
}

bool PS4_SYSV_ABI ReadRepeatingProgram(void* context, VAddr address, void* output, u64 size) {
    auto& program = *static_cast<RepeatingProgram*>(context);
    ++program.calls;
    if (address < program.address || size != sizeof(program.word) ||
        address - program.address >= MaxFetchShaderProgramSize) {
        return false;
    }
    std::memcpy(output, &program.word, sizeof(program.word));
    return true;
}

template <size_t Size>
void StorePointer(std::array<u32, Size>& user_data, u32 index, const u32* pointer) {
    const VAddr address = reinterpret_cast<VAddr>(pointer);
    user_data[index] = static_cast<u32>(address);
    user_data[index + 1] = static_cast<u32>(address >> 32);
}

Info MakeFetchInfo(std::span<const u32> user_data) {
    Info info{};
    info.has_fetch_shader = true;
    info.fetch_shader_sgpr_base = 0;
    info.user_data = user_data;
    info.srt_info.flattened_bufsize_dw = user_data.size();
    info.flattened_ud_buf.assign(user_data.begin(), user_data.end());
    return info;
}

TEST(FetchShaderCapture, RefusedCompileReadNeverFallsBackToLiveGuestCode) {
    const std::array<u32, 1> live_program{
        SOP1(OpcodeSOP1::S_SETPC_B64, SOperand7::S0, SOperand8::S0).Get(),
    };
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    StorePointer(user_data, 0, live_program.data());
    auto info = MakeFetchInfo(user_data);
    u32 read_calls{};
    CompileResourceCapture capture{{RefuseGuestProgram, &read_calls}};
    const auto reader = capture.Reader();
    info.UseCompileCaptureReader(reader);

    EXPECT_FALSE(ParseFetchShader(info).has_value());
    EXPECT_EQ(read_calls, 1U);
    EXPECT_FALSE(reader.Valid());
}

TEST(FetchShaderCapture, ParsesAndReplaysOwnedCompileReaderBytesInsteadOfLiveCode) {
    const std::array<u32, 1> live_program{
        SOP1(OpcodeSOP1::S_SETPC_B64, SOperand7::S0, SOperand8::S0).Get(),
    };
    std::array<u32, 2> captured_program{
        VOP2(OpcodeVOP2::V_ADD_I32, VOperand8::V0, SOperand9::S6, VOperand8::V0).Get(),
        SOP1(OpcodeSOP1::S_SETPC_B64, SOperand7::S0, SOperand8::S0).Get(),
    };
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    StorePointer(user_data, 0, live_program.data());
    auto info = MakeFetchInfo(user_data);
    GuestProgram memory{reinterpret_cast<VAddr>(live_program.data()), captured_program};
    CompileResourceCapture capture{{ReadGuestProgram, &memory}};
    info.UseCompileCaptureReader(capture.Reader());

    const auto selected = ParseFetchShader(info);
    EXPECT_EQ(memory.calls, 2U);
    EXPECT_EQ(memory.read_addresses[0], memory.address);
    EXPECT_EQ(memory.read_addresses[1], memory.address + sizeof(u32));
    EXPECT_EQ(memory.read_sizes[0], sizeof(u32));
    EXPECT_EQ(memory.read_sizes[1], sizeof(u32));
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(selected->size, 2U * sizeof(u32));
    EXPECT_EQ(selected->vertex_offset_sgpr, 6);

    auto snapshot = capture.Finish(info);
    ASSERT_TRUE(snapshot.valid);
    captured_program[0] = captured_program[1];
    auto replay = MakeFetchInfo(user_data);
    replay.UseCompileResourceSnapshot(std::move(snapshot));

    const auto replayed = ParseFetchShader(replay);
    ASSERT_TRUE(replayed.has_value());
    EXPECT_EQ(replayed->size, 2U * sizeof(u32));
    EXPECT_EQ(replayed->vertex_offset_sgpr, 6);
    EXPECT_EQ(memory.calls, 2U);
}

TEST(FetchShaderCapture, DumpCopyIsOwnedBoundedAndUsesTheCompileReader) {
    const std::array<u32, 1> live_program{
        SOP1(OpcodeSOP1::S_SETPC_B64, SOperand7::S0, SOperand8::S0).Get(),
    };
    std::array<u32, 2> captured_program{0x11223344U, 0x55667788U};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    StorePointer(user_data, 0, live_program.data());
    auto info = MakeFetchInfo(user_data);
    GuestProgram memory{reinterpret_cast<VAddr>(live_program.data()), captured_program};
    CompileResourceCapture capture{{ReadGuestProgram, &memory}};
    info.UseCompileCaptureReader(capture.Reader());

    const auto owned = ReadFetchShaderProgram(info, 0, captured_program.size() * sizeof(u32));
    ASSERT_TRUE(owned.has_value());
    EXPECT_EQ(memory.calls, 1U);
    EXPECT_EQ(memory.read_sizes[0], captured_program.size() * sizeof(u32));
    EXPECT_EQ(std::memcmp(owned->data(), captured_program.data(), owned->size()), 0);

    captured_program.fill(0U);
    EXPECT_EQ((*owned)[0], 0x44U);
    EXPECT_FALSE(ReadFetchShaderProgram(info, 0, MaxFetchShaderProgramSize + 1).has_value());
    EXPECT_EQ(memory.calls, 1U);
}

TEST(FetchShaderCapture, MissingTerminatorStopsAtTheHardProgramCeiling) {
    constexpr VAddr GuestAddress = 0x285200000ULL;
    constexpr u32 SNop = 0xBF800000U;
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    user_data[0] = static_cast<u32>(GuestAddress);
    user_data[1] = static_cast<u32>(GuestAddress >> 32);
    auto info = MakeFetchInfo(user_data);
    RepeatingProgram memory{GuestAddress, SNop};
    CompileResourceCapture capture{{ReadRepeatingProgram, &memory}};
    const auto reader = capture.Reader();
    info.UseCompileCaptureReader(reader);

    EXPECT_FALSE(ParseFetchShader(info).has_value());
    EXPECT_EQ(memory.calls, MaxFetchShaderProgramSize / sizeof(u32));
    EXPECT_TRUE(reader.Valid());
}

TEST(FetchShaderCapture, TranslateProgramNonfatallyRejectsARefusedFetchCapture) {
    constexpr VAddr GuestAddress = 0x285200000ULL;
    const std::array<u32, 2> shader_program{
        SOP1(OpcodeSOP1::S_SWAPPC_B64, SOperand7::S0, SOperand8::S0).Get(),
        0xBF810000U, // s_endpgm
    };
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    user_data[0] = static_cast<u32>(GuestAddress);
    user_data[1] = static_cast<u32>(GuestAddress >> 32);
    auto info = MakeFetchInfo(user_data);
    info.stage = Stage::Vertex;
    info.l_stage = LogicalStage::Vertex;
    u32 read_calls{};
    CompileResourceCapture capture{{RefuseGuestProgram, &read_calls}};
    const auto reader = capture.Reader();
    info.UseCompileCaptureReader(reader);
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Vertex);
    runtime_info.num_user_data = NUM_USER_DATA_REGS;
    Profile profile{};
    profile.subgroup_size = 64;
    Pools pools{};

    const auto program =
        TranslateProgram(shader_program, pools, info, runtime_info, profile, reader);
    EXPECT_FALSE(program.blocks.empty());
    EXPECT_EQ(read_calls, 1U);
    EXPECT_FALSE(reader.Valid());
    EXPECT_FALSE(info.translation_failed);
}

} // namespace
} // namespace Shader::Gcn
