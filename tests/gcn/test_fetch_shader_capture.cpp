// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>

#include <gtest/gtest.h>

#include "instructions.hpp"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"

namespace Shader::Gcn {
namespace {

struct GuestProgram {
    VAddr address{};
    std::span<const u32> words;
    u32 calls{};
    bool failed{};
};

struct RepeatingProgram {
    VAddr address{};
    u32 word{};
    u32 calls{};
    bool failed{};
};

struct GuestBytes {
    VAddr address{};
    std::span<const std::byte> bytes;
    u32 calls{};
    bool failed{};
};

bool PS4_SYSV_ABI ReadGuestProgram(void* context, VAddr address, void* output, u64 size) {
    auto& program = *static_cast<GuestProgram*>(context);
    ++program.calls;
    const auto bytes = std::as_bytes(program.words);
    if (address < program.address || size > bytes.size() ||
        address - program.address > bytes.size() - size) {
        program.failed = true;
        return false;
    }
    std::memcpy(output, bytes.data() + (address - program.address), size);
    return true;
}

bool PS4_SYSV_ABI RefuseGuestProgram(void* context, VAddr, void*, u64) {
    auto& failed = *static_cast<bool*>(context);
    failed = true;
    return false;
}

bool PS4_SYSV_ABI ReadRepeatingProgram(void* context, VAddr address, void* output, u64 size) {
    auto& program = *static_cast<RepeatingProgram*>(context);
    ++program.calls;
    if (address < program.address || size != sizeof(program.word) ||
        address - program.address >= MaxFetchShaderProgramSize) {
        program.failed = true;
        return false;
    }
    std::memcpy(output, &program.word, sizeof(program.word));
    return true;
}

bool PS4_SYSV_ABI ReadGuestBytes(void* context, VAddr address, void* output, u64 size) {
    auto& memory = *static_cast<GuestBytes*>(context);
    ++memory.calls;
    if (address < memory.address || size > memory.bytes.size() ||
        address - memory.address > memory.bytes.size() - size) {
        memory.failed = true;
        return false;
    }
    std::memcpy(output, memory.bytes.data() + (address - memory.address), size);
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
    bool failed{};
    const SrtMemoryReader reader{RefuseGuestProgram, &failed, &failed};

    EXPECT_FALSE(ParseFetchShader(info, reader).has_value());
    EXPECT_TRUE(failed);
    EXPECT_FALSE(reader.Valid());
}

TEST(FetchShaderCapture, ParsesReaderBytesInsteadOfLiveGuestCode) {
    const std::array<u32, 1> live_program{
        SOP1(OpcodeSOP1::S_SETPC_B64, SOperand7::S0, SOperand8::S0).Get(),
    };
    const std::array<u32, 2> captured_program{
        VOP2(OpcodeVOP2::V_ADD_I32, VOperand8::V0, SOperand9::S6, VOperand8::V0).Get(),
        SOP1(OpcodeSOP1::S_SETPC_B64, SOperand7::S0, SOperand8::S0).Get(),
    };
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    StorePointer(user_data, 0, live_program.data());
    auto info = MakeFetchInfo(user_data);
    GuestProgram memory{reinterpret_cast<VAddr>(live_program.data()), captured_program};
    const SrtMemoryReader reader{ReadGuestProgram, &memory, &memory.failed};

    const auto selected = ParseFetchShader(info, reader);
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(selected->size, 2U * sizeof(u32));
    EXPECT_EQ(selected->vertex_offset_sgpr, 6);
    EXPECT_EQ(memory.calls, 2U);
    EXPECT_TRUE(reader.Valid());
}

TEST(FetchShaderCapture, UsesTrailingSmrdLiteralAsDwordOffset) {
    // GCN 1.1 encodes a wide SMRD offset as OFFSET=0xff, IMM=0, followed by
    // a literal dword. The literal is a dword offset, not a byte offset and
    // not the 0xff sentinel. Keep this as a fetch-shader test because the
    // resulting V# lookup is what consumes the decoded offset.
    constexpr u32 LiteralOffset = 0x20;
    constexpr u64 BufferLoadFormatXyzw = 0x80020400e00c2000ULL;
    const std::array<u32, 5> program{
        // s_load_dwordx4 s[8:11], s[2:3], literal
        0xc08402ffU,
        LiteralOffset,
        static_cast<u32>(BufferLoadFormatXyzw),
        static_cast<u32>(BufferLoadFormatXyzw >> 32),
        SOP1(OpcodeSOP1::S_SETPC_B64, SOperand7::S0, SOperand8::S0).Get(),
    };
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    StorePointer(user_data, 0, program.data());
    auto info = MakeFetchInfo(user_data);
    GuestProgram memory{reinterpret_cast<VAddr>(program.data()), program};
    const SrtMemoryReader reader{ReadGuestProgram, &memory, &memory.failed};

    const auto selected = ParseFetchShader(info, reader);
    ASSERT_TRUE(selected.has_value());
    ASSERT_EQ(selected->attributes.size(), 1U);
    EXPECT_EQ(selected->attributes[0].sgpr_base, 2);
    EXPECT_EQ(selected->attributes[0].dword_offset, LiteralOffset);
    EXPECT_EQ(selected->attributes[0].dest_vgpr, 4);
    EXPECT_EQ(selected->attributes[0].num_elements, 4);
    EXPECT_EQ(selected->size, program.size() * sizeof(u32));
    EXPECT_TRUE(reader.Valid());
    EXPECT_FALSE(memory.failed);
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
    const SrtMemoryReader reader{ReadGuestProgram, &memory, &memory.failed};

    const auto owned =
        ReadFetchShaderProgram(info, reader, 0, captured_program.size() * sizeof(u32));
    ASSERT_TRUE(owned.has_value());
    EXPECT_EQ(std::memcmp(owned->data(), captured_program.data(), owned->size()), 0);
    captured_program.fill(0U);
    EXPECT_EQ((*owned)[0], 0x44U);
    EXPECT_FALSE(
        ReadFetchShaderProgram(info, reader, 0, MaxFetchShaderProgramSize + 1).has_value());
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
    const SrtMemoryReader reader{ReadRepeatingProgram, &memory, &memory.failed};

    EXPECT_FALSE(ParseFetchShader(info, reader).has_value());
    EXPECT_EQ(memory.calls, MaxFetchShaderProgramSize / sizeof(u32));
    EXPECT_TRUE(reader.Valid());
}

TEST(FetchShaderCapture, VertexBuffersAreOwnedFromTheCompileSnapshot) {
    std::array<AmdGpu::Buffer, 2> live_buffers{};
    live_buffers[0].base_address = 0x111100;
    live_buffers[0].stride = 16;
    live_buffers[0].num_records = 4;
    live_buffers[1].base_address = 0x222200;
    live_buffers[1].stride = 32;
    live_buffers[1].num_records = 8;

    auto captured_buffers = live_buffers;
    captured_buffers[0].base_address = 0xAAAA00;
    captured_buffers[0].stride = 48;
    captured_buffers[1].base_address = 0xBBBB00;
    captured_buffers[1].stride = 64;

    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    StorePointer(user_data, 0, reinterpret_cast<const u32*>(live_buffers.data()));
    auto info = MakeFetchInfo(user_data);
    FetchShaderData fetch{};
    fetch.attributes.push_back(VertexAttribute{.semantic = 3, .sgpr_base = 0, .dword_offset = 0});
    fetch.attributes.push_back(VertexAttribute{.semantic = 7, .sgpr_base = 0, .dword_offset = 4});

    GuestBytes memory{
        reinterpret_cast<VAddr>(live_buffers.data()), std::as_bytes(std::span{captured_buffers})};
    const SrtMemoryReader reader{ReadGuestBytes, &memory, &memory.failed};

    ASSERT_TRUE(CaptureVertexBuffers(info, fetch, reader));
    ASSERT_EQ(info.captured_vertex_buffers.size(), 2U);
    EXPECT_EQ(info.captured_vertex_buffers[0], captured_buffers[0]);
    EXPECT_EQ(info.captured_vertex_buffers[1], captured_buffers[1]);
    EXPECT_NE(info.captured_vertex_buffers[0], live_buffers[0]);
    EXPECT_EQ(memory.calls, 2U);
    EXPECT_TRUE(reader.Valid());
}

TEST(FetchShaderCapture, RefusedVertexBufferCapturePublishesNothing) {
    std::array<AmdGpu::Buffer, 1> live_buffers{};
    live_buffers[0].base_address = 0x111100;
    live_buffers[0].stride = 16;
    live_buffers[0].num_records = 4;
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    StorePointer(user_data, 0, reinterpret_cast<const u32*>(live_buffers.data()));
    auto info = MakeFetchInfo(user_data);
    info.captured_vertex_buffers.push_back(live_buffers[0]);
    FetchShaderData fetch{};
    fetch.attributes.push_back(VertexAttribute{.semantic = 0, .sgpr_base = 0, .dword_offset = 0});
    bool failed{};
    const SrtMemoryReader reader{RefuseGuestProgram, &failed, &failed};

    EXPECT_FALSE(CaptureVertexBuffers(info, fetch, reader));
    EXPECT_TRUE(failed);
    EXPECT_FALSE(reader.Valid());
    ASSERT_EQ(info.captured_vertex_buffers.size(), 1U);
    EXPECT_EQ(info.captured_vertex_buffers[0], live_buffers[0]);
}

} // namespace
} // namespace Shader::Gcn
