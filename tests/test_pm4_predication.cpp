// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/amdgpu/pm4_predication.h"

namespace AmdGpu {
namespace {

TEST(Pm4Predication, DecodesFlagsFirstPacketLayout) {
    constexpr u32 flags = 1u << 8 | 1u << 12 | 3u << 16;
    constexpr std::array payload{flags, 0x23456780u, 0x12u};

    const auto packet = DecodeSetPredication(payload);

    ASSERT_TRUE(packet);
    EXPECT_EQ(packet->address, 0x1223456780u);
    EXPECT_EQ(packet->condition, 1u);
    EXPECT_TRUE(packet->wait_for_results);
    EXPECT_EQ(packet->operation, 3u);
}

TEST(Pm4Predication, DecodesAddressFirstPacketLayout) {
    constexpr u32 flags_and_high = 0x12u | 3u << 16;
    constexpr std::array payload{0x23456780u, flags_and_high};

    const auto packet = DecodeSetPredication(payload);

    ASSERT_TRUE(packet);
    EXPECT_EQ(packet->address, 0x1223456780u);
    EXPECT_EQ(packet->condition, 0u);
    EXPECT_FALSE(packet->wait_for_results);
    EXPECT_EQ(packet->operation, 3u);
}

TEST(Pm4Predication, RejectsTruncatedPacket) {
    constexpr std::array<u32, 1> payload{0};
    EXPECT_FALSE(DecodeSetPredication(payload));
}

TEST(Pm4Predication, DisableOperationClearsSkipping) {
    DecodedPredication packet{.operation = 0};
    EXPECT_EQ(EvaluatePredication(packet, 123), false);
}

TEST(Pm4Predication, BooleanOperationImplementsBothConditions) {
    DecodedPredication packet{.condition = 0, .operation = 3};
    EXPECT_EQ(EvaluatePredication(packet, 0), false);
    EXPECT_EQ(EvaluatePredication(packet, 1), true);

    packet.condition = 1;
    EXPECT_EQ(EvaluatePredication(packet, 0), true);
    EXPECT_EQ(EvaluatePredication(packet, 1), false);
}

TEST(Pm4Predication, RejectsUnsupportedOperationAndCondition) {
    EXPECT_FALSE(EvaluatePredication(DecodedPredication{.operation = 2}, 0));
    EXPECT_FALSE(EvaluatePredication(DecodedPredication{.condition = 2, .operation = 3}, 0));
}

TEST(Pm4Predication, OnlyPredicateEnabledPacketsObserveSkipState) {
    EXPECT_FALSE(ShouldSkipPm4Packet(false, false));
    EXPECT_FALSE(ShouldSkipPm4Packet(false, true));
    EXPECT_FALSE(ShouldSkipPm4Packet(true, false));
    EXPECT_TRUE(ShouldSkipPm4Packet(true, true));
}

} // namespace
} // namespace AmdGpu
