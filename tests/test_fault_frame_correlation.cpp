// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/fault_frame_correlation.h"

namespace VideoCore {
namespace {

TEST(FaultFrameCorrelation, SupportsTheRequiredLongDiagnosticWindow) {
    EXPECT_GE(FaultFrameCorrelation::HardMaxFrames, 1700);
}

TEST(FaultFrameCorrelation, DisabledAndOutsideFramesDoNotMutateReducerState) {
    FaultFrameCorrelation disabled{{.enabled = false, .first_frame = 10, .frame_count = 3}};
    disabled.RecordFlip(10, 1000);
    const auto disabled_stamp = disabled.CaptureStamp();
    disabled.ObserveBatch(disabled_stamp, std::array<u64, 1>{7}, false);
    EXPECT_EQ(disabled.GetCoverage(), FaultFrameCorrelationCoverage{});
    EXPECT_TRUE(disabled.Finish().empty());

    FaultFrameCorrelation bounded{{.enabled = true, .first_frame = 10, .frame_count = 3}};
    bounded.RecordFlip(9, 900);
    const auto before_stamp = bounded.CaptureStamp();
    bounded.ObserveBatch(before_stamp, std::array<u64, 1>{7}, false);
    bounded.RecordFlip(13, 1300);
    const auto after_stamp = bounded.CaptureStamp();
    bounded.ObserveBatch(after_stamp, std::array<u64, 1>{8}, false);
    EXPECT_EQ(bounded.GetCoverage(), FaultFrameCorrelationCoverage{});
    EXPECT_TRUE(bounded.Finish().empty());
}

TEST(FaultFrameCorrelation, AggregatesBatchesAndSortsAndDeduplicatesPrivatePageIds) {
    FaultFrameCorrelation diagnostic{
        {.enabled = true, .first_frame = 10, .frame_count = 3, .page_cap = 8}};
    diagnostic.RecordFlip(10, 1000);
    const auto stamp = diagnostic.CaptureStamp();
    diagnostic.ObserveBatch(stamp, std::array<u64, 3>{5, 3, 5}, false);
    diagnostic.ObserveBatch(stamp, std::array<u64, 2>{4, 3}, false);

    const auto observations = diagnostic.Finish();
    ASSERT_EQ(observations.size(), 1);
    EXPECT_EQ(observations[0].frame_sequence, 10);
    EXPECT_EQ(observations[0].process_time_us, 1000);
    EXPECT_EQ(observations[0].page_count, 3);
    EXPECT_EQ(observations[0].batch_count, 2);
    EXPECT_EQ(observations[0].status, FaultFrameCorrelationStatus::Complete);
    EXPECT_EQ(observations[0].loss, FaultFrameCorrelationLoss::None);

    const auto coverage = diagnostic.GetCoverage();
    EXPECT_EQ(coverage.selected_frames, 1);
    EXPECT_EQ(coverage.complete_frames, 1);
    EXPECT_EQ(coverage.total_batches, 2);
    EXPECT_EQ(coverage.total_unique_pages, 3);
}

TEST(FaultFrameCorrelation, ReportsStableChangeAndExactAbaWithoutExposingIds) {
    FaultFrameCorrelation diagnostic{
        {.enabled = true, .first_frame = 20, .frame_count = 3, .page_cap = 8}};
    const std::array<std::array<u64, 2>, 3> pages{{{{1, 2}}, {{3, 3}}, {{2, 1}}}};
    for (u64 i = 0; i < pages.size(); ++i) {
        diagnostic.RecordFlip(20 + i, 2000 + i * 16);
        diagnostic.ObserveBatch(diagnostic.CaptureStamp(), pages[i], false);
    }

    const auto observations = diagnostic.Finish();
    ASSERT_EQ(observations.size(), 3);
    EXPECT_FALSE(observations[0].stable);
    EXPECT_FALSE(observations[0].changed);
    EXPECT_FALSE(observations[1].stable);
    EXPECT_TRUE(observations[1].changed);
    EXPECT_FALSE(observations[1].exact_aba);
    EXPECT_FALSE(observations[2].stable);
    EXPECT_TRUE(observations[2].changed);
    EXPECT_TRUE(observations[2].exact_aba);
    EXPECT_EQ(observations[2].page_count, 2);

    const auto coverage = diagnostic.GetCoverage();
    EXPECT_EQ(coverage.changed_frames, 2);
    EXPECT_EQ(coverage.exact_aba_frames, 1);
}

TEST(FaultFrameCorrelation, IncompleteFramesAndSequenceGapsSuppressAba) {
    FaultFrameCorrelation incomplete{
        {.enabled = true, .first_frame = 30, .frame_count = 3, .page_cap = 1}};
    incomplete.RecordFlip(30, 3000);
    incomplete.ObserveBatch(incomplete.CaptureStamp(), std::array<u64, 1>{1}, false);
    incomplete.RecordFlip(31, 3016);
    incomplete.ObserveBatch(incomplete.CaptureStamp(), std::array<u64, 2>{2, 3}, true);
    incomplete.RecordFlip(32, 3032);
    incomplete.ObserveBatch(incomplete.CaptureStamp(), std::array<u64, 1>{1}, false);

    const auto incomplete_observations = incomplete.Finish();
    ASSERT_EQ(incomplete_observations.size(), 3);
    EXPECT_EQ(incomplete_observations[1].status, FaultFrameCorrelationStatus::Incomplete);
    EXPECT_NE(incomplete_observations[1].loss & FaultFrameCorrelationLoss::DownloadOverflow,
              FaultFrameCorrelationLoss::None);
    EXPECT_NE(incomplete_observations[1].loss & FaultFrameCorrelationLoss::PageCapacity,
              FaultFrameCorrelationLoss::None);
    EXPECT_FALSE(incomplete_observations[2].exact_aba);

    FaultFrameCorrelation gap{
        {.enabled = true, .first_frame = 40, .frame_count = 4, .page_cap = 8}};
    gap.RecordFlip(40, 4000);
    gap.ObserveBatch(gap.CaptureStamp(), std::array<u64, 1>{1}, false);
    gap.RecordFlip(42, 4032);
    gap.ObserveBatch(gap.CaptureStamp(), std::array<u64, 1>{2}, false);
    gap.RecordFlip(43, 4048);
    gap.ObserveBatch(gap.CaptureStamp(), std::array<u64, 1>{1}, false);

    const auto gap_observations = gap.Finish();
    ASSERT_EQ(gap_observations.size(), 3);
    EXPECT_EQ(gap_observations[1].status, FaultFrameCorrelationStatus::Gap);
    EXPECT_FALSE(gap_observations[2].exact_aba);
    EXPECT_EQ(gap.GetCoverage().gap_frames, 1);
}

TEST(FaultFrameCorrelation, EmitsSelectedFramesWithoutBatchesAndBoundsConfiguration) {
    FaultFrameCorrelation diagnostic{{.enabled = true,
                                      .first_frame = 1,
                                      .frame_count = FaultFrameCorrelation::HardMaxFrames + 10,
                                      .page_cap = FaultFrameCorrelation::HardMaxPagesPerFrame + 10}};
    diagnostic.RecordFlip(1, 100);
    diagnostic.RecordFlip(2, 116);

    const auto observations = diagnostic.Finish();
    ASSERT_EQ(observations.size(), 2);
    EXPECT_EQ(observations[0].status, FaultFrameCorrelationStatus::NoBatches);
    EXPECT_EQ(observations[1].status, FaultFrameCorrelationStatus::NoBatches);
    EXPECT_EQ(diagnostic.GetCoverage().no_batch_frames, 2);
    EXPECT_EQ(diagnostic.GetConfiguration().frame_count, FaultFrameCorrelation::HardMaxFrames);
    EXPECT_EQ(diagnostic.GetConfiguration().page_cap,
              FaultFrameCorrelation::HardMaxPagesPerFrame);
}

TEST(FaultFrameCorrelation, ReportingCannotFreezeBeforeDeferredCallbacksDrain) {
    FaultFrameCorrelationReportGate gate;
    EXPECT_FALSE(gate.ClaimAfterDrain());
    gate.MarkDeferredCallbacksDrained();
    EXPECT_TRUE(gate.ClaimAfterDrain());
    EXPECT_FALSE(gate.ClaimAfterDrain());
}

} // namespace
} // namespace VideoCore
