// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/amdgpu/pm4_predication.h"

namespace AmdGpu {
namespace {

TEST(Pm4Predication, DecodesLiverpoolAddressAndControlFields) {
    constexpr u32 control_and_high =
        0x12u | (1u << 8) | (1u << 12) | (1u << 16) | (1u << 31);
    constexpr std::array payload{0x2345678fu, control_and_high};

    const auto packet = DecodeLiverpoolSetPredication(payload);

    ASSERT_TRUE(packet);
    EXPECT_EQ(packet->address, 0x1223456780u);
    EXPECT_TRUE(packet->draw_if_visible);
    EXPECT_FALSE(packet->wait_for_results);
    EXPECT_EQ(packet->operation, PredicationOperation::Zpass);
    EXPECT_TRUE(packet->continue_predicate);
}

TEST(Pm4Predication, ZeroHintMeansWaitAndClearDisablesThePredicate) {
    constexpr std::array payload{0u, 0u};

    const auto packet = DecodeLiverpoolSetPredication(payload);

    ASSERT_TRUE(packet);
    EXPECT_TRUE(packet->wait_for_results);
    EXPECT_EQ(packet->operation, PredicationOperation::Clear);
    EXPECT_FALSE(packet->continue_predicate);
}

TEST(Pm4Predication, PreservesReservedOperationsForFailOpenHandling) {
    constexpr std::array payload{0x1000u, 3u << 16};

    const auto packet = DecodeLiverpoolSetPredication(payload);

    ASSERT_TRUE(packet);
    EXPECT_EQ(packet->operation, PredicationOperation::Reserved3);
}

TEST(Pm4Predication, ControlAndReservedBitsNeverLeakIntoTheAddress) {
    constexpr u32 address_high = 0x5au;
    constexpr u32 non_address_bits = 0x7ff7ff00u;
    constexpr std::array payload{0xabcdef0fu, address_high | non_address_bits};

    const auto packet = DecodeLiverpoolSetPredication(payload);

    ASSERT_TRUE(packet);
    EXPECT_EQ(packet->address, 0x5aabcdef00u);
}

TEST(Pm4Predication, RejectsTruncatedPacket) {
    constexpr std::array<u32, 1> payload{0};
    EXPECT_FALSE(DecodeLiverpoolSetPredication(payload));
}

} // namespace
} // namespace AmdGpu
