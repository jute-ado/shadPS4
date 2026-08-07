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
    diagnostic.ObserveRejectedPacket(ShortBody.size());
    constexpr std::array<u32, 4> Body{0, 0, 0, 7};
    for (u32 i = 0; i < CondExecDiagnostic::MaxRecords + 3; ++i) {
        const auto packet = DecodeGfx7CondExec(Body);
        ASSERT_TRUE(packet.has_value());
        diagnostic.Observe(*packet, (i & 1U) != 0);
    }

    const auto report = diagnostic.Report();
    EXPECT_EQ(report.total_packets, CondExecDiagnostic::MaxRecords + 4);
    EXPECT_EQ(report.gfx7_packets, CondExecDiagnostic::MaxRecords + 3);
    EXPECT_EQ(report.decode_rejections, 1U);
    EXPECT_EQ(report.retained_records, CondExecDiagnostic::MaxRecords);
    EXPECT_EQ(report.truncated_records, 3U);
    EXPECT_EQ(report.false_conditions, (CondExecDiagnostic::MaxRecords + 4) / 2);
    EXPECT_EQ(report.layout_mismatches, CondExecDiagnostic::MaxRecords + 3);
}

TEST(CondExecPacket, CorrelatesFullWordConditionsWithQueryAndPredicationState) {
    constexpr std::array<u32, 4> Body{0x1004, 0, 0, 7};
    const auto packet = DecodeGfx7CondExec(Body);
    ASSERT_TRUE(packet.has_value());

    CondExecDiagnostic diagnostic;
    diagnostic.ObserveQueryDump(0x1000, 16, 2);
    diagnostic.ObservePredicatedPacket();
    diagnostic.ObserveSetPredication();
    diagnostic.Observe(*packet, 0x00000100, true);

    const auto report = diagnostic.TakeFrameReport();
    EXPECT_EQ(report.total_packets, 1U);
    EXPECT_EQ(report.gfx7_packets, 1U);
    EXPECT_EQ(report.nonzero_exec_counts, 1U);
    EXPECT_EQ(report.full_word_nonzero_low_byte_zero, 1U);
    EXPECT_EQ(report.gpu_modified_conditions, 1U);
    EXPECT_EQ(report.query_dumps, 1U);
    EXPECT_EQ(report.query_condition_overlaps, 1U);
    EXPECT_EQ(report.predicated_packets, 1U);
    EXPECT_EQ(report.set_predication_packets, 1U);

    EXPECT_EQ(diagnostic.TakeFrameReport().total_packets, 0U);

    constexpr std::array<u32, 4> PaddingBody{0x100c, 0, 0, 7};
    constexpr std::array<u32, 4> NextLaneBody{0x1014, 0, 0, 7};
    const auto padding_packet = DecodeGfx7CondExec(PaddingBody);
    const auto next_lane_packet = DecodeGfx7CondExec(NextLaneBody);
    ASSERT_TRUE(padding_packet.has_value());
    ASSERT_TRUE(next_lane_packet.has_value());
    diagnostic.Observe(*padding_packet, 1, false);
    diagnostic.Observe(*next_lane_packet, 1, false);

    const auto next_frame = diagnostic.TakeFrameReport();
    EXPECT_EQ(next_frame.total_packets, 2U);
    EXPECT_EQ(next_frame.query_dumps, 0U);
    EXPECT_EQ(next_frame.query_condition_overlaps, 1U);
}

TEST(CondExecPacket, ReportsBaselineAndBoundedFrameTupleChanges) {
    CondExecFrameChangeTracker changes;
    CondExecDiagnosticReport report{};

    EXPECT_TRUE(changes.ShouldReport(report));
    EXPECT_FALSE(changes.ShouldReport(report));
    report.total_packets = 1;
    EXPECT_TRUE(changes.ShouldReport(report));
    EXPECT_FALSE(changes.ShouldReport(report));

    for (u32 i = 2; i < CondExecFrameChangeTracker::MaxRecords; ++i) {
        report.total_packets = i;
        EXPECT_TRUE(changes.ShouldReport(report));
    }
    report.total_packets = CondExecFrameChangeTracker::MaxRecords;
    EXPECT_FALSE(changes.ShouldReport(report));
    EXPECT_EQ(changes.ReportedRecords(), CondExecFrameChangeTracker::MaxRecords);
    EXPECT_EQ(changes.TruncatedRecords(), 1U);
}

} // namespace
} // namespace AmdGpu
