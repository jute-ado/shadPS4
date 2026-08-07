// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/indirect_argument_dataflow.h"

namespace AmdGpu {
namespace {

TEST(IndirectArgumentDataflow, DetectsQueryDumpOverlapInEitherObservationOrder) {
    IndirectArgumentDataflowTracker tracker;

    tracker.ObserveQueryDump(0x1000, 16, 3);
    tracker.ObserveIndirect({.address = 0x1010, .size = 8});
    tracker.ObserveIndirect({.address = 0x2000, .size = 8});
    tracker.ObserveQueryDump(0x2000, 16, 1);

    const auto report = tracker.TakeFrameReport();
    EXPECT_EQ(report.indirect_draws, 2U);
    EXPECT_EQ(report.query_dumps, 2U);
    EXPECT_EQ(report.query_overlap_pairs, 2U);
}

TEST(IndirectArgumentDataflow, TracksCpuVisibleChangesAcrossFrames) {
    IndirectArgumentDataflowTracker tracker;
    constexpr IndirectArgumentObservation first{
        .address = 0x3000,
        .size = 20,
        .content_hash = 0x1234,
        .content_hash_valid = true,
        .zero_command = false,
    };

    tracker.ObserveIndirect(first);
    auto report = tracker.TakeFrameReport();
    EXPECT_EQ(report.cpu_visible_draws, 1U);
    EXPECT_EQ(report.first_observed_ranges, 1U);
    EXPECT_EQ(report.content_changes, 0U);

    tracker.ObserveIndirect(first);
    tracker.ObserveIndirect({
        .address = first.address,
        .size = first.size,
        .content_hash = 0x5678,
        .content_hash_valid = true,
        .zero_command = true,
    });
    report = tracker.TakeFrameReport();
    EXPECT_EQ(report.cpu_visible_draws, 2U);
    EXPECT_EQ(report.first_observed_ranges, 0U);
    EXPECT_EQ(report.content_changes, 1U);
    EXPECT_EQ(report.zero_commands, 1U);
}

TEST(IndirectArgumentDataflow, SeparatesGpuModifiedArgumentsFromCpuHashes) {
    IndirectArgumentDataflowTracker tracker;

    tracker.ObserveIndirect({
        .address = 0x4000,
        .size = 20,
        .gpu_modified = true,
    });

    const auto report = tracker.TakeFrameReport();
    EXPECT_EQ(report.indirect_draws, 1U);
    EXPECT_EQ(report.gpu_modified_draws, 1U);
    EXPECT_EQ(report.cpu_visible_draws, 0U);
    EXPECT_EQ(report.first_observed_ranges, 0U);
}

TEST(IndirectArgumentDataflow, BoundsPerFrameRangesWithoutLosingCounts) {
    IndirectArgumentDataflowTracker tracker;

    for (u32 i = 0; i < IndirectArgumentDataflowTracker::MaxFrameRanges + 1; ++i) {
        tracker.ObserveIndirect({.address = 0x10000 + i * 0x100, .size = 20});
    }

    const auto report = tracker.TakeFrameReport();
    EXPECT_EQ(report.indirect_draws, IndirectArgumentDataflowTracker::MaxFrameRanges + 1);
    EXPECT_EQ(report.truncated_indirect_ranges, 1U);
}

} // namespace
} // namespace AmdGpu
