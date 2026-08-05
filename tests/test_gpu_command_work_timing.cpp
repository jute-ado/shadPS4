// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/gpu_command_work_timing.h"

namespace {

using AmdGpu::GpuCommandWorkCategory;
using AmdGpu::GpuCommandWorkTiming;

TEST(GpuCommandWorkTiming, RequiresExactOptInValue) {
    EXPECT_FALSE(AmdGpu::GpuCommandWorkTimingRequested(nullptr));
    EXPECT_FALSE(AmdGpu::GpuCommandWorkTimingRequested(""));
    EXPECT_TRUE(AmdGpu::GpuCommandWorkTimingRequested("1"));
    EXPECT_FALSE(AmdGpu::GpuCommandWorkTimingRequested("0"));
    EXPECT_FALSE(AmdGpu::GpuCommandWorkTimingRequested("true"));
    EXPECT_FALSE(AmdGpu::GpuCommandWorkTimingRequested("10"));
}

TEST(GpuCommandWorkTiming, ReportsAndResetsBoundedCategoryTotals) {
    constexpr std::uint64_t StartNanoseconds = 1'000;
    GpuCommandWorkTiming timing{StartNanoseconds};

    timing.RecordPacket(7);
    timing.RecordPacket(13);
    timing.Record(GpuCommandWorkCategory::Resume, 100);
    timing.Record(GpuCommandWorkCategory::Draw, 40);
    timing.Record(GpuCommandWorkCategory::Dispatch, 10);
    timing.Record(GpuCommandWorkCategory::Transfer, 5);
    timing.Record(GpuCommandWorkCategory::Sync, 7);

    EXPECT_FALSE(timing.ShouldReport(StartNanoseconds +
                                     AmdGpu::GpuCommandWorkReportIntervalNanoseconds - 1));
    EXPECT_TRUE(
        timing.ShouldReport(StartNanoseconds + AmdGpu::GpuCommandWorkReportIntervalNanoseconds));

    const auto snapshot =
        timing.TakeSnapshot(StartNanoseconds + AmdGpu::GpuCommandWorkReportIntervalNanoseconds);
    EXPECT_EQ(snapshot.packet_count, 2);
    EXPECT_EQ(snapshot.packet_dwords, 20);
    EXPECT_EQ(snapshot.At(GpuCommandWorkCategory::Resume).calls, 1);
    EXPECT_EQ(snapshot.At(GpuCommandWorkCategory::Resume).nanoseconds, 100);
    EXPECT_EQ(snapshot.At(GpuCommandWorkCategory::Draw).nanoseconds, 40);
    EXPECT_EQ(snapshot.At(GpuCommandWorkCategory::Dispatch).nanoseconds, 10);
    EXPECT_EQ(snapshot.At(GpuCommandWorkCategory::Transfer).nanoseconds, 5);
    EXPECT_EQ(snapshot.At(GpuCommandWorkCategory::Sync).nanoseconds, 7);
    EXPECT_EQ(snapshot.UnclassifiedResumeNanoseconds(), 38);

    const auto reset =
        timing.TakeSnapshot(StartNanoseconds + 2 * AmdGpu::GpuCommandWorkReportIntervalNanoseconds);
    EXPECT_EQ(reset.packet_count, 0);
    EXPECT_EQ(reset.packet_dwords, 0);
    EXPECT_EQ(reset.At(GpuCommandWorkCategory::Resume).calls, 0);
    EXPECT_EQ(reset.At(GpuCommandWorkCategory::Resume).nanoseconds, 0);
    EXPECT_EQ(reset.UnclassifiedResumeNanoseconds(), 0);
}

TEST(GpuCommandWorkTiming, UnclassifiedResumeTimeCannotUnderflow) {
    GpuCommandWorkTiming timing{0};
    timing.Record(GpuCommandWorkCategory::Resume, 5);
    timing.Record(GpuCommandWorkCategory::Draw, 7);
    timing.Record(GpuCommandWorkCategory::Dispatch, 11);

    const auto snapshot = timing.TakeSnapshot(AmdGpu::GpuCommandWorkReportIntervalNanoseconds);
    EXPECT_EQ(snapshot.UnclassifiedResumeNanoseconds(), 0);
}

} // namespace
