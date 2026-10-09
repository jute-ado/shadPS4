// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include "video_core/amdgpu/pm4_lod_stats.h"

namespace AmdGpu {
namespace {

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{input}, {}};
}

std::string ReadSiblingSource(const char* source_path, const char* sibling_name) {
    const auto path = std::filesystem::path{source_path}.replace_filename(sibling_name);
    return ReadSource(path.string().c_str());
}

constexpr std::array<u32, 3> ValidPayload{
    0x1020u,
    0x09210001u,
    0x000bfc00u,
};

TEST(Pm4LodStats, DecodesAddressSizeAndControl) {
    const auto packet = DecodeLodStatsPacket(ValidPayload);

    ASSERT_TRUE(packet);
    EXPECT_EQ(packet->address, 0x09210000ull);
    EXPECT_EQ(packet->size, 0x1020u);
    EXPECT_EQ(packet->control, 0x000bfc00u);
}

TEST(Pm4LodStats, MasksAlignmentAndReservedControlBitsFromAddress) {
    constexpr std::array<u32, 3> payload{
        0x40u,
        0xdeadbeffu,
        0x0000005au,
    };

    const auto packet = DecodeLodStatsPacket(payload);

    ASSERT_TRUE(packet);
    EXPECT_EQ(packet->address, 0x5adeadbec0ull);
}

TEST(Pm4LodStats, RejectsTruncatedAndOverlongPayloads) {
    constexpr std::array<u32, 2> truncated{0x40u, 0x1000u};
    constexpr std::array<u32, 4> overlong{0x40u, 0x1000u, 0u, 0u};

    EXPECT_FALSE(DecodeLodStatsPacket(truncated));
    EXPECT_FALSE(DecodeLodStatsPacket(overlong));
    constexpr std::array<u32, 0> empty{};
    EXPECT_FALSE(DecodeLodStatsPacket(empty));
}

TEST(Pm4LodStats, RejectsZeroAddressAndZeroSize) {
    constexpr std::array<u32, 3> zero_address{0x40u, 0u, 0u};
    constexpr std::array<u32, 3> zero_size{0u, 0x1000u, 0u};

    EXPECT_FALSE(DecodeLodStatsPacket(zero_address));
    EXPECT_FALSE(DecodeLodStatsPacket(zero_size));
}

TEST(Pm4LodStats, AcceptsRangeEndingAtGpuAddressLimit) {
    constexpr std::array<u32, 3> last_range{
        0x40u,
        0xffffffc0u,
        0x000000ffu,
    };

    const auto packet = DecodeLodStatsPacket(last_range);

    ASSERT_TRUE(packet);
    EXPECT_EQ(packet->address, 0xffffffffc0ull);
    EXPECT_EQ(packet->size, 0x40u);
}

TEST(Pm4LodStats, RejectsRangeOverflowPastGpuAddressLimit) {
    constexpr std::array<u32, 3> overflowing_range{
        0x41u,
        0xffffffc0u,
        0x000000ffu,
    };

    EXPECT_FALSE(DecodeLodStatsPacket(overflowing_range));
}

TEST(Pm4LodStats, KeepsControlBitsOutOfAddress) {
    constexpr std::array<u32, 3> payload{
        0x40u,
        0x0000103fu,
        0xabcd005au,
    };

    const auto packet = DecodeLodStatsPacket(payload);

    ASSERT_TRUE(packet);
    EXPECT_EQ(packet->address, 0x5a00001000ull);
    EXPECT_EQ(packet->control, 0xabcd005au);
}

TEST(Pm4LodStats, Type3HeaderPreservesMaximumBodyCountForRealOpcodes) {
    const auto source = ReadSiblingSource(SHADPS4_LIVERPOOL_SOURCE_PATH, "pm4_cmds.h");
    ASSERT_FALSE(source.empty());
    const auto begin = source.find("u32 NumWords() const", source.find("union PM4Type3Header"));
    const auto end = source.find("};", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    // Type-3 COUNT is a 14-bit N - 1 body count. The all-ones value therefore describes
    // 0x4000 body dwords; only the local zero-body NOP sentinel may map it back to zero.
    EXPECT_NE(body.find("opcode == PM4ItOpcode::Nop"), std::string::npos);
    EXPECT_NE(body.find("return 0"), std::string::npos);
    EXPECT_NE(body.find("count + 1"), std::string::npos);
}

TEST(Pm4LodStats, RuntimeHandlerDoesNotDereferenceOrFabricateGuestReport) {
    const auto source = ReadSource(SHADPS4_LIVERPOOL_SOURCE_PATH);
    ASSERT_FALSE(source.empty());
    const auto begin = source.find("case PM4ItOpcode::GetLodStats:");
    const auto end = source.find("case PM4ItOpcode::CondExec:", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    // The PM4 packet's report format and completion protocol are not established by the parser
    // alone. Until they are, the handler must remain non-mutating: never reinterpret the guest
    // VAddr as a host pointer and never publish a fabricated "complete" zero report.
    EXPECT_EQ(body.find("reinterpret_cast"), std::string::npos);
    EXPECT_EQ(body.find("InitializeLodStatsReport"), std::string::npos);
    EXPECT_EQ(body.find("std::memcpy"), std::string::npos);
    EXPECT_NE(body.find("IT_GET_LOD_STATS"), std::string::npos);
}

TEST(Pm4LodStats, RuntimeHandlerValidatesPacketBeforeSyntaxOnlyDecode) {
    const auto source = ReadSource(SHADPS4_LIVERPOOL_SOURCE_PATH);
    ASSERT_FALSE(source.empty());
    const auto begin = source.find("case PM4ItOpcode::GetLodStats:");
    const auto end = source.find("case PM4ItOpcode::CondExec:", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    // Syntax-only decoding is safe only after the packet body has been proven to contain all
    // payload dwords. Report execution remains intentionally deferred until its schema and
    // completion ordering are established.
    EXPECT_NE(body.find("NumWords"), std::string::npos);
    EXPECT_NE(body.find("dcb.size()"), std::string::npos);
    EXPECT_NE(body.find("DecodeLodStatsPacket"), std::string::npos);
    EXPECT_NE(body.find("syntax-only"), std::string::npos);
}

TEST(Pm4LodStats, ZeroBodyNopSentinelIsCheckedBeforePayloadRead) {
    const auto source = ReadSource(SHADPS4_LIVERPOOL_SOURCE_PATH);
    ASSERT_FALSE(source.empty());
    const auto begin = source.find("case PM4ItOpcode::Nop:");
    const auto end = source.find("case PM4ItOpcode::SetBase:", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    const auto guard = body.find("nop_count == 0x3fffu && nop->header.NumWords() == 0");
    const auto payload = body.find("nop->data_block[0]");
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(payload, std::string::npos);
    EXPECT_LT(guard, payload);
}

TEST(Pm4LodStats, ConstantEngineUsesNormalizedType3PacketSize) {
    const auto source = ReadSource(SHADPS4_LIVERPOOL_SOURCE_PATH);
    ASSERT_FALSE(source.empty());
    const auto begin = source.find("Liverpool::Task Liverpool::ProcessCeUpdate(");
    const auto end = source.find("Liverpool::Task Liverpool::ProcessGraphics(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto process_ce = source.substr(begin, end - begin);

    const auto packet_begin = process_ce.find("const auto packet_words =");
    ASSERT_NE(packet_begin, std::string::npos);
    const auto packet_end = process_ce.find(';', packet_begin);
    ASSERT_NE(packet_end, std::string::npos);
    const auto packet_size = process_ce.substr(packet_begin, packet_end - packet_begin);

    // PM4Type3Header::NumWords() normalizes the zero-body trailing NOP sentinel. CE must add only
    // the header word after that normalization instead of reinterpreting the raw COUNT field.
    EXPECT_NE(packet_size.find("header->type3.NumWords()"), std::string::npos);
    EXPECT_NE(packet_size.find("+ 1"), std::string::npos);
    EXPECT_EQ(packet_size.find("header->type3.count.Value()"), std::string::npos);
}

} // namespace
} // namespace AmdGpu
