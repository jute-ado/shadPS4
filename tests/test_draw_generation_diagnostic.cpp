// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/draw_generation_diagnostic.h"

using AmdGpu::DrawGenerationDiagnostic;
using AmdGpu::DrawIssueKind;

namespace {

void RecordFrame(DrawGenerationDiagnostic& diagnostic, std::initializer_list<u64> signatures) {
    u32 ordinal = 0;
    for (const u64 signature : signatures) {
        const auto kind = ordinal++ == 0 ? DrawIssueKind::DirectIndexed
                                         : DrawIssueKind::IndirectIndexed;
        diagnostic.ObserveDraw(kind, signature);
    }
}

} // namespace

TEST(DrawGenerationDiagnostic, StableOrderedDrawsDoNotReportChanges) {
    DrawGenerationDiagnostic diagnostic;

    RecordFrame(diagnostic, {10, 20, 30});
    const auto first = diagnostic.TakeSnapshot();
    EXPECT_EQ(first.draws, 3);
    EXPECT_EQ(first.direct_draws, 1);
    EXPECT_EQ(first.indirect_draws, 2);
    EXPECT_FALSE(first.has_previous);

    RecordFrame(diagnostic, {10, 20, 30});
    const auto second = diagnostic.TakeSnapshot();
    EXPECT_TRUE(second.has_previous);
    EXPECT_FALSE(second.count_changed);
    EXPECT_EQ(second.changed_from_previous, 0);
    EXPECT_EQ(second.exact_aba_return_draws, 0);

    RecordFrame(diagnostic, {10, 20, 30});
    const auto third = diagnostic.TakeSnapshot();
    EXPECT_EQ(third.changed_from_previous, 0);
    EXPECT_EQ(third.exact_aba_return_draws, 0);
}

TEST(DrawGenerationDiagnostic, DetectsSameCountOrdinalSubstitutionAndExactAbaReturn) {
    DrawGenerationDiagnostic diagnostic;

    RecordFrame(diagnostic, {10, 20, 30});
    const auto baseline = diagnostic.TakeSnapshot();
    EXPECT_EQ(baseline.sequence, 1);
    RecordFrame(diagnostic, {10, 99, 30});
    const auto changed = diagnostic.TakeSnapshot();
    EXPECT_FALSE(changed.count_changed);
    EXPECT_EQ(changed.changed_from_previous, 1);
    ASSERT_EQ(changed.reported_changed_from_previous, 1);
    EXPECT_EQ(changed.first_changed_from_previous_ordinals[0], 1);

    RecordFrame(diagnostic, {10, 20, 30});
    const auto returned = diagnostic.TakeSnapshot();
    EXPECT_EQ(returned.changed_from_previous, 1);
    EXPECT_EQ(returned.exact_aba_return_draws, 1);
    EXPECT_EQ(returned.aba_middle_sequence, 2);
    ASSERT_EQ(returned.reported_exact_aba_return_draws, 1);
    EXPECT_EQ(returned.first_exact_aba_return_ordinals[0], 1);
}

TEST(DrawGenerationDiagnostic, ReportsCountChangesAndSubmissionMutation) {
    DrawGenerationDiagnostic diagnostic;

    diagnostic.ObserveSubmission(/*submitted_dcb=*/100, /*parsed_dcb=*/100,
                                 /*submitted_ccb=*/200, /*parsed_ccb=*/200);
    RecordFrame(diagnostic, {10, 20});
    const auto first = diagnostic.TakeSnapshot();
    EXPECT_EQ(first.submissions, 1);
    EXPECT_EQ(first.mutated_submissions, 0);

    diagnostic.ObserveSubmission(/*submitted_dcb=*/100, /*parsed_dcb=*/101,
                                 /*submitted_ccb=*/200, /*parsed_ccb=*/200);
    RecordFrame(diagnostic, {10, 20, 30});
    const auto second = diagnostic.TakeSnapshot();
    EXPECT_TRUE(second.count_changed);
    EXPECT_EQ(second.changed_from_previous, 1);
    EXPECT_EQ(second.submissions, 1);
    EXPECT_EQ(second.mutated_submissions, 1);
}

TEST(DrawGenerationDiagnostic, BoundsHistoryAndReportsTruncation) {
    DrawGenerationDiagnostic diagnostic;

    for (u32 draw = 0; draw < DrawGenerationDiagnostic::MaxDrawsPerFrame + 3; ++draw) {
        diagnostic.ObserveDraw(DrawIssueKind::DirectNonIndexed, draw);
    }
    for (u32 submit = 0; submit < DrawGenerationDiagnostic::MaxSubmissionsPerFrame + 2; ++submit) {
        diagnostic.ObserveSubmission(submit, submit, submit, submit);
    }

    const auto snapshot = diagnostic.TakeSnapshot();
    EXPECT_EQ(snapshot.draws, DrawGenerationDiagnostic::MaxDrawsPerFrame);
    EXPECT_EQ(snapshot.truncated_draws, 3);
    EXPECT_EQ(snapshot.submissions, DrawGenerationDiagnostic::MaxSubmissionsPerFrame);
    EXPECT_EQ(snapshot.truncated_submissions, 2);
}

TEST(DrawGenerationDiagnostic, SignatureBuilderIsOrderSensitive) {
    AmdGpu::DrawGenerationSignature forward;
    forward.Add(10u);
    forward.Add(20u);

    AmdGpu::DrawGenerationSignature reverse;
    reverse.Add(20u);
    reverse.Add(10u);

    EXPECT_NE(forward.Value(), reverse.Value());
    EXPECT_EQ(AmdGpu::HashCommandWords(std::array<u32, 3>{1, 2, 3}),
              AmdGpu::HashCommandWords(std::array<u32, 3>{1, 2, 3}));
    EXPECT_NE(AmdGpu::HashCommandWords(std::array<u32, 3>{1, 2, 3}),
              AmdGpu::HashCommandWords(std::array<u32, 3>{1, 9, 3}));
}
