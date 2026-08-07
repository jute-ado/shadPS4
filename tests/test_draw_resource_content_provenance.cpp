// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <array>

#include "video_core/amdgpu/draw_resource_content_provenance_diagnostic.h"

using AmdGpu::ClassifyDrawResourceUpload;
using AmdGpu::DrawResourceContentProvenanceDiagnostic;
using AmdGpu::DrawResourceContentProbeMode;
using AmdGpu::DrawResourceUploadProvenance;

TEST(DrawResourceContentProvenanceDiagnostic, ClassifiesUploadReceipts) {
    EXPECT_EQ(ClassifyDrawResourceUpload(1, 1, 1), DrawResourceUploadProvenance::Coherent);
    EXPECT_EQ(ClassifyDrawResourceUpload(1, 1, 2),
              DrawResourceUploadProvenance::ConcurrentWriterStagedBefore);
    EXPECT_EQ(ClassifyDrawResourceUpload(1, 2, 2),
              DrawResourceUploadProvenance::ConcurrentWriterStagedAfter);
    EXPECT_EQ(ClassifyDrawResourceUpload(1, 3, 2),
              DrawResourceUploadProvenance::ConcurrentWriterOther);
    EXPECT_EQ(ClassifyDrawResourceUpload(1, 2, 1),
              DrawResourceUploadProvenance::StagedMismatch);
}

TEST(DrawResourceContentProvenanceDiagnostic, FingerprintsContentForEqualityOnly) {
    DrawResourceContentProvenanceDiagnostic diagnostic{/*report_limit=*/1};
    const std::array<u8, 4> first{1, 2, 3, 4};
    const std::array<u8, 4> same{1, 2, 3, 4};
    const std::array<u8, 4> different{1, 2, 3, 5};

    EXPECT_EQ(diagnostic.FingerprintBytes(first.data(), first.size()),
              diagnostic.FingerprintBytes(same.data(), same.size()));
    EXPECT_NE(diagnostic.FingerprintBytes(first.data(), first.size()),
              diagnostic.FingerprintBytes(different.data(), different.size()));
}

TEST(DrawResourceContentProvenanceDiagnostic, ReportsCoherentContentChangeAndAba) {
    DrawResourceContentProvenanceDiagnostic diagnostic{/*report_limit=*/3};

    const auto record = [&](u64 token) {
        diagnostic.BeginDraw();
        diagnostic.RecordCpuUpload(token, token, token, 64);
        diagnostic.EndDraw();
        return diagnostic.TakeSnapshot();
    };

    const auto first = record(1);
    EXPECT_EQ(first.observations, 1);
    EXPECT_EQ(first.bytes_probed, 64);
    EXPECT_EQ(first.coherent_changed_resources, 0);
    EXPECT_EQ(first.coherent_content_aba_resources, 0);

    const auto middle = record(2);
    EXPECT_EQ(middle.coherent_changed_resources, 1);
    EXPECT_EQ(middle.coherent_content_aba_resources, 0);

    const auto returned = record(1);
    EXPECT_EQ(returned.coherent_changed_resources, 1);
    EXPECT_EQ(returned.coherent_content_aba_resources, 1);
    ASSERT_EQ(returned.reported_coherent_content_aba_resources, 1);
    EXPECT_EQ(returned.first_coherent_content_aba_resources[0].draw_ordinal, 0);
    EXPECT_EQ(returned.first_coherent_content_aba_resources[0].resource_ordinal, 0);
}

TEST(DrawResourceContentProvenanceDiagnostic, SeparatesWriterOverlapAndTornStaging) {
    DrawResourceContentProvenanceDiagnostic diagnostic{/*report_limit=*/1};
    diagnostic.BeginDraw();
    diagnostic.RecordCpuUpload(1, 1, 2, 16);
    diagnostic.RecordCpuUpload(1, 2, 2, 16);
    diagnostic.RecordCpuUpload(1, 3, 2, 16);
    diagnostic.RecordCpuUpload(1, 2, 1, 16);
    diagnostic.EndDraw();

    const auto snapshot = diagnostic.TakeSnapshot();
    EXPECT_EQ(snapshot.concurrent_write_resources, 3);
    EXPECT_EQ(snapshot.endpoint_capture_resources, 2);
    EXPECT_EQ(snapshot.staged_mismatch_resources, 1);
}

TEST(DrawResourceContentProvenanceDiagnostic, MarksResidentAndTruncatedContentUnknown) {
    DrawResourceContentProvenanceDiagnostic diagnostic{/*report_limit=*/1};
    diagnostic.BeginDraw();
    diagnostic.RecordResidentUnobserved();
    diagnostic.RecordTruncated(4096);
    diagnostic.AbortDraw();

    diagnostic.BeginDraw();
    diagnostic.RecordResidentUnobserved();
    diagnostic.RecordTruncated(8192);
    diagnostic.EndDraw();

    const auto snapshot = diagnostic.TakeSnapshot();
    EXPECT_EQ(snapshot.draws, 1);
    EXPECT_EQ(snapshot.observations, 2);
    EXPECT_EQ(snapshot.resident_unobserved_resources, 1);
    EXPECT_EQ(snapshot.truncated_resources, 1);
    EXPECT_EQ(snapshot.truncated_bytes, 8192);
}

TEST(DrawResourceContentProvenanceDiagnostic, CapturesOnlyConfiguredFrameWindow) {
    DrawResourceContentProvenanceDiagnostic diagnostic{/*report_limit=*/4};
    diagnostic.ConfigureCaptureWindow(/*start=*/2, /*count=*/1);

    diagnostic.BeginDraw();
    EXPECT_FALSE(diagnostic.IsDrawActive());
    const auto before = diagnostic.TakeSnapshot();
    EXPECT_FALSE(before.should_report);
    EXPECT_EQ(before.sequence, 1);

    diagnostic.BeginDraw();
    ASSERT_TRUE(diagnostic.IsDrawActive());
    diagnostic.RecordCpuUpload(1, 1, 1, 64);
    diagnostic.EndDraw();
    const auto inside = diagnostic.TakeSnapshot();
    EXPECT_TRUE(inside.should_report);
    EXPECT_EQ(inside.sequence, 2);
    EXPECT_EQ(inside.observations, 1);

    diagnostic.BeginDraw();
    EXPECT_FALSE(diagnostic.IsDrawActive());
    const auto after = diagnostic.TakeSnapshot();
    EXPECT_FALSE(after.should_report);
    EXPECT_EQ(after.sequence, 3);
}

TEST(DrawResourceContentProvenanceDiagnostic, ReportsStagedOnlyContentAba) {
    DrawResourceContentProvenanceDiagnostic diagnostic{/*report_limit=*/3};

    const auto record = [&](u64 token) {
        diagnostic.BeginDraw();
        diagnostic.RecordStagedUpload(token, 64);
        diagnostic.EndDraw();
        return diagnostic.TakeSnapshot();
    };

    EXPECT_EQ(record(1).staged_content_aba_resources, 0);
    const auto middle = record(2);
    EXPECT_EQ(middle.staged_changed_resources, 1);
    EXPECT_EQ(middle.staged_content_aba_resources, 0);

    const auto returned = record(1);
    EXPECT_EQ(returned.staged_changed_resources, 1);
    EXPECT_EQ(returned.staged_content_aba_resources, 1);
    ASSERT_EQ(returned.reported_staged_content_aba_resources, 1);
    EXPECT_EQ(returned.first_staged_content_aba_resources[0].draw_ordinal, 0);
    EXPECT_EQ(returned.first_staged_content_aba_resources[0].resource_ordinal, 0);
}

TEST(DrawResourceContentProvenanceDiagnostic, ReportsSourceSnapshotContentAba) {
    DrawResourceContentProvenanceDiagnostic diagnostic{/*report_limit=*/3};

    const auto record = [&](u64 token) {
        diagnostic.BeginDraw();
        diagnostic.RecordSourceSnapshot(token, 64);
        diagnostic.EndDraw();
        return diagnostic.TakeSnapshot();
    };

    EXPECT_EQ(record(1).source_content_aba_resources, 0);
    EXPECT_EQ(record(2).source_changed_resources, 1);
    const auto returned = record(1);
    EXPECT_EQ(returned.source_changed_resources, 1);
    EXPECT_EQ(returned.source_content_aba_resources, 1);
    ASSERT_EQ(returned.reported_source_content_aba_resources, 1);
    EXPECT_EQ(returned.first_source_content_aba_resources[0].draw_ordinal, 0);
    EXPECT_EQ(returned.first_source_content_aba_resources[0].resource_ordinal, 0);
}

TEST(DrawResourceContentProvenanceDiagnostic, SelectsExplicitStagedOnlyProbeMode) {
    DrawResourceContentProvenanceDiagnostic diagnostic{/*report_limit=*/1};
    EXPECT_EQ(diagnostic.GetProbeMode(), DrawResourceContentProbeMode::FullProvenance);

    diagnostic.ConfigureProbeMode(DrawResourceContentProbeMode::StagedOnly);
    EXPECT_EQ(diagnostic.GetProbeMode(), DrawResourceContentProbeMode::StagedOnly);

    diagnostic.ConfigureProbeMode(DrawResourceContentProbeMode::SourceSnapshot);
    EXPECT_EQ(diagnostic.GetProbeMode(), DrawResourceContentProbeMode::SourceSnapshot);
}

TEST(DrawResourceContentProvenanceDiagnostic, EnforcesConfiguredPerFrameProbeBudget) {
    DrawResourceContentProvenanceDiagnostic diagnostic{/*report_limit=*/2};
    diagnostic.ConfigureProbeByteLimit(64);

    diagnostic.BeginDraw();
    ASSERT_TRUE(diagnostic.CanProbeCpuUpload(64));
    diagnostic.RecordStagedUpload(1, 64);
    EXPECT_FALSE(diagnostic.CanProbeCpuUpload(1));
    diagnostic.EndDraw();
    EXPECT_EQ(diagnostic.TakeSnapshot().bytes_probed, 64);

    diagnostic.BeginDraw();
    EXPECT_TRUE(diagnostic.CanProbeCpuUpload(64));
}

TEST(DrawResourceContentProvenanceDiagnostic, SkipsConfiguredBytesBeforeProbing) {
    DrawResourceContentProvenanceDiagnostic diagnostic{/*report_limit=*/1};
    diagnostic.ConfigureProbeByteSkip(64);

    diagnostic.BeginDraw();
    ASSERT_TRUE(diagnostic.ShouldSkipCpuUpload(64));
    diagnostic.RecordSkipped(64);
    EXPECT_FALSE(diagnostic.ShouldSkipCpuUpload(16));
    ASSERT_TRUE(diagnostic.CanProbeCpuUpload(16));
    diagnostic.RecordStagedUpload(1, 16);
    diagnostic.EndDraw();

    const auto snapshot = diagnostic.TakeSnapshot();
    EXPECT_EQ(snapshot.skipped_resources, 1);
    EXPECT_EQ(snapshot.skipped_bytes, 64);
    EXPECT_EQ(snapshot.bytes_probed, 16);
}
