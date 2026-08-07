// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/draw_resource_content_provenance_diagnostic.h"

using AmdGpu::ClassifyDrawResourceUpload;
using AmdGpu::DrawResourceContentProvenanceDiagnostic;
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
