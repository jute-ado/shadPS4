// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/amdgpu/indirect_args_fingerprint_diagnostic.h"

using AmdGpu::IndirectArgsFingerprintDiagnostic;
using AmdGpu::IndirectArgsKind;

TEST(IndirectArgsFingerprintDiagnostic, IdentifiesWhichOrderedInvocationChanged) {
    IndirectArgsFingerprintDiagnostic diagnostic{/*report_limit=*/3};
    std::array<u32, 8> first{10, 1, 0, 0, 20, 1, 4, 0};
    std::array<u32, 5> second{30, 1, 8, 0, 0};

    diagnostic.Record(IndirectArgsKind::Draw, first.data(), /*stride=*/16, /*count=*/2,
                      /*command_size=*/16);
    diagnostic.Record(IndirectArgsKind::DrawIndexed, second.data(), /*stride=*/20, /*count=*/1,
                      /*command_size=*/20);
    const auto baseline = diagnostic.TakeSnapshot();

    EXPECT_TRUE(baseline.should_report);
    EXPECT_EQ(baseline.sequence, 1);
    EXPECT_EQ(baseline.invocations, 2);
    EXPECT_EQ(baseline.argument_records, 3);
    EXPECT_EQ(baseline.bytes_hashed, 52);
    EXPECT_EQ(baseline.changed_invocation_mask, 0b11);
    EXPECT_FALSE(baseline.matches_previous_frame);

    diagnostic.Record(IndirectArgsKind::Draw, first.data(), 16, 2, 16);
    diagnostic.Record(IndirectArgsKind::DrawIndexed, second.data(), 20, 1, 20);
    const auto unchanged = diagnostic.TakeSnapshot();

    EXPECT_EQ(unchanged.combined_hash, baseline.combined_hash);
    EXPECT_EQ(unchanged.changed_invocation_mask, 0);
    EXPECT_TRUE(unchanged.matches_previous_frame);

    second[0] = 31;
    diagnostic.Record(IndirectArgsKind::Draw, first.data(), 16, 2, 16);
    diagnostic.Record(IndirectArgsKind::DrawIndexed, second.data(), 20, 1, 20);
    const auto changed = diagnostic.TakeSnapshot();

    EXPECT_NE(changed.combined_hash, unchanged.combined_hash);
    EXPECT_EQ(changed.changed_invocation_mask, 0b10);
    EXPECT_FALSE(changed.matches_previous_frame);
}

TEST(IndirectArgsFingerprintDiagnostic, BoundsInvocationsAndArgumentBytes) {
    IndirectArgsFingerprintDiagnostic diagnostic{/*report_limit=*/1};
    std::array<u32, 4> args{1, 2, 3, 4};

    for (u32 i = 0; i < IndirectArgsFingerprintDiagnostic::MaxInvocations + 2; ++i) {
        diagnostic.Record(IndirectArgsKind::Draw, args.data(), 16, 1, 16);
    }
    const auto snapshot = diagnostic.TakeSnapshot();

    EXPECT_EQ(snapshot.invocations, IndirectArgsFingerprintDiagnostic::MaxInvocations);
    EXPECT_EQ(snapshot.truncated_invocations, 2);
    EXPECT_FALSE(diagnostic.TakeSnapshot().should_report);
}
