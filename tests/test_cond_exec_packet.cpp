// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/amdgpu/cond_exec_packet.h"

namespace AmdGpu {
namespace {

TEST(CondExecPacket, DecodesGfx7CountAfterReservedControlWord) {
    constexpr std::array<u32, 4> Body{
        0x1234567b,
        0x00000002,
        0x00000000,
        0xffffc123,
    };

    const auto packet = DecodeGfx7CondExec(Body);
    ASSERT_TRUE(packet.has_value());
    EXPECT_EQ(packet->address, 0x0000000212345678ULL);
    EXPECT_EQ(packet->exec_count, 0x123U);
    EXPECT_FALSE(packet->reserved_control_nonzero);
    EXPECT_FALSE(packet->legacy_count_matches);
}

TEST(CondExecPacket, RejectsNonGfx7BodyAndBoundsDiagnosticRecords) {
    constexpr std::array<u32, 3> ShortBody{};
    EXPECT_FALSE(DecodeGfx7CondExec(ShortBody).has_value());

    CondExecDiagnostic diagnostic;
    constexpr std::array<u32, 4> Body{0, 0, 0, 7};
    for (u32 i = 0; i < CondExecDiagnostic::MaxRecords + 3; ++i) {
        const auto packet = DecodeGfx7CondExec(Body);
        ASSERT_TRUE(packet.has_value());
        diagnostic.Observe(*packet, (i & 1U) != 0);
    }

    const auto report = diagnostic.Report();
    EXPECT_EQ(report.total_packets, CondExecDiagnostic::MaxRecords + 3);
    EXPECT_EQ(report.retained_records, CondExecDiagnostic::MaxRecords);
    EXPECT_EQ(report.truncated_records, 3U);
    EXPECT_EQ(report.false_conditions, (CondExecDiagnostic::MaxRecords + 4) / 2);
    EXPECT_EQ(report.layout_mismatches, CondExecDiagnostic::MaxRecords + 3);
}

} // namespace
} // namespace AmdGpu
