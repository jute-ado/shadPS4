// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
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

constexpr std::array<u32, 4> ValidPayload{
    0x1020u,
    0x09210001u,
    0x00000000u,
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
    constexpr std::array<u32, 4> payload{
        0x40u,
        0xdeadbeffu,
        0x0000005au,
        0xabcd005au,
    };

    const auto packet = DecodeLodStatsPacket(payload);

    ASSERT_TRUE(packet);
    EXPECT_EQ(packet->address, 0x5adeadbec0ull);
}

TEST(Pm4LodStats, RejectsTruncatedAndOverlongPayloads) {
    constexpr std::array<u32, 3> truncated{0x40u, 0x1000u, 0u};
    constexpr std::array<u32, 5> overlong{0x40u, 0x1000u, 0u, 0u, 0u};

    EXPECT_FALSE(DecodeLodStatsPacket(truncated));
    EXPECT_FALSE(DecodeLodStatsPacket(overlong));
    constexpr std::array<u32, 0> empty{};
    EXPECT_FALSE(DecodeLodStatsPacket(empty));
}

TEST(Pm4LodStats, RejectsZeroAddressAndZeroSize) {
    constexpr std::array<u32, 4> zero_address{0x40u, 0u, 0u, 0u};
    constexpr std::array<u32, 4> zero_size{0u, 0x1000u, 0u, 0u};

    EXPECT_FALSE(DecodeLodStatsPacket(zero_address));
    EXPECT_FALSE(DecodeLodStatsPacket(zero_size));
}

TEST(Pm4LodStats, AcceptsRangeEndingAtGpuAddressLimit) {
    constexpr std::array<u32, 4> last_range{
        0x40u,
        0xffffffc0u,
        0x000000ffu,
        0u,
    };

    const auto packet = DecodeLodStatsPacket(last_range);

    ASSERT_TRUE(packet);
    EXPECT_EQ(packet->address, 0xffffffffc0ull);
    EXPECT_EQ(packet->size, 0x40u);
}

TEST(Pm4LodStats, RejectsRangeOverflowPastGpuAddressLimit) {
    constexpr std::array<u32, 4> overflowing_range{
        0x41u,
        0xffffffc0u,
        0x000000ffu,
        0u,
    };

    EXPECT_FALSE(DecodeLodStatsPacket(overflowing_range));
}

TEST(Pm4LodStats, RejectsAddressBitsOutsideTheFortyBitGpuRange) {
    constexpr std::array<u32, 4> invalid_high_bits{
        0x40u,
        0x00001000u,
        0x00000100u,
        0u,
    };

    EXPECT_FALSE(DecodeLodStatsPacket(invalid_high_bits));
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

} // namespace
} // namespace AmdGpu
