// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <gtest/gtest.h>

#include "video_core/amdgpu/pm4_lod_stats.h"

TEST(Pm4LodStats, DecodesLiverpoolReportPacket) {
    constexpr std::array<u32, 3> payload{0x1020, 0x09210001, 0x000bfc00};

    const AmdGpu::LodStatsReport report = AmdGpu::DecodeLodStatsReport(payload);

    EXPECT_EQ(report.size, 0x1020);
    EXPECT_EQ(report.address, 0x09210000);
    EXPECT_EQ(report.control, 0x000bfc00);
}

TEST(Pm4LodStats, InitializesCompletedEmptyReport) {
    std::array<u32, 8> storage;
    storage.fill(0xdeadbeef);

    AmdGpu::InitializeLodStatsReport(std::as_writable_bytes(std::span{storage}));

    EXPECT_EQ(storage[0], 1);
    EXPECT_TRUE(
        std::ranges::all_of(std::span{storage}.subspan(1), [](u32 value) { return value == 0; }));
}
