// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <gtest/gtest.h>

#include "video_core/amdgpu/pm4_lod_stats.h"

TEST(Pm4LodStats, DecodesObservedShadowRequestWithoutTreatingControlAsAddress) {
    constexpr std::array<u32, 3> payload{0x1020, 0x0909fe81, 0x000bfc00};

    const auto request = AmdGpu::DecodeLodStatsRequest(payload);

    EXPECT_EQ(request.size, 0x1020);
    EXPECT_EQ(request.address, 0x0909fe80);
    EXPECT_EQ(request.reset_count, 0xff);
    EXPECT_FALSE(request.force_reset);
    EXPECT_TRUE(request.report_and_reset);
    EXPECT_EQ(request.reporting_interval, 0);
}

TEST(Pm4LodStats, InitializesCompletedEmptyReport) {
    std::array<u32, 8> storage;
    storage.fill(0xdeadbeef);

    AmdGpu::InitializeLodStatsReport(std::as_writable_bytes(std::span{storage}));

    EXPECT_EQ(storage[0], 1U);
    EXPECT_TRUE(
        std::ranges::all_of(std::span{storage}.subspan(1), [](const u32 value) { return value == 0; }));
}

TEST(Pm4LodStats, BoundsCompletedReportWritesToTheHardwareReportCapacity) {
    EXPECT_FALSE(AmdGpu::IsLodStatsReportSizeSupported(0));
    EXPECT_FALSE(AmdGpu::IsLodStatsReportSizeSupported(3));
    EXPECT_TRUE(AmdGpu::IsLodStatsReportSizeSupported(4));
    EXPECT_TRUE(AmdGpu::IsLodStatsReportSizeSupported(0x1020));
    EXPECT_FALSE(AmdGpu::IsLodStatsReportSizeSupported(0x1024));
}

TEST(Pm4LodStats, SummarizesExistingReportWithoutMutatingIt) {
    constexpr std::array<u32, 8> report{0, 4, 0, 7, 0, 0, 9, 0};

    const auto summary = AmdGpu::SummarizeLodStatsReport(report);

    EXPECT_EQ(summary.nonzero_dwords, 3);
    EXPECT_EQ(summary.first_nonzero_dword, 1);
    EXPECT_EQ(summary.first_nonzero_value, 4);
    EXPECT_EQ(report[3], 7);
}

TEST(Pm4LodStats, SamplesStartupAndLaterPowerOfTwoRequests) {
    for (u32 request = 0; request < 8; ++request) {
        EXPECT_TRUE(AmdGpu::ShouldInspectLodStats(request));
    }
    EXPECT_TRUE(AmdGpu::ShouldInspectLodStats(8));
    EXPECT_FALSE(AmdGpu::ShouldInspectLodStats(9));
    EXPECT_TRUE(AmdGpu::ShouldInspectLodStats(16));
    EXPECT_FALSE(AmdGpu::ShouldInspectLodStats(17));
    EXPECT_TRUE(AmdGpu::ShouldInspectLodStats(1024));
}

TEST(Pm4LodStats, ReadWatchCoversEveryPageTouchedByUnalignedReport) {
    constexpr auto watch = AmdGpu::MakeLodStatsReadWatch(0x0909fe80, 0x1020, 0x1000);

    EXPECT_EQ(watch.page_address, 0x0909f000);
    EXPECT_EQ(watch.page_size, 0x2000);
    EXPECT_TRUE(watch.Contains(0x0909fe80));
    EXPECT_TRUE(watch.Contains(0x090a0e9f));
    EXPECT_FALSE(watch.Contains(0x0909efff));
    EXPECT_FALSE(watch.Contains(0x090a0ea0));
}
