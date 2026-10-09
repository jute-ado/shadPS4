// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <limits>
#include <span>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/srt_snapshot_composition_receipt.h"

namespace VideoCore {
namespace {

using Route = SrtSnapshotCompositionRoute;

constexpr SrtSnapshotCompositionReceiptConfig GpuConfiguration{
    .snapshot_address = 0x1000,
    .snapshot_size = 64,
    .focus_address = 0x1008,
    .focus_size = 40,
    .route = Route::GpuComposed,
    .overlay_count = 2,
    .overlay_bytes = 48,
};

TEST(SrtSnapshotCompositionReceipt, CapturesBoundedIdenticalGpuComposition) {
    std::array<u8, 64> cpu{};
    for (size_t i = 0; i < cpu.size(); ++i) {
        cpu[i] = static_cast<u8>(i);
    }
    const auto composed = cpu;

    const auto receipt = CaptureSrtSnapshotCompositionReceipt(GpuConfiguration, cpu, composed);

    ASSERT_TRUE(receipt);
    EXPECT_EQ(receipt->snapshot_address, 0x1000U);
    EXPECT_EQ(receipt->snapshot_size, 64U);
    EXPECT_EQ(receipt->focus_address, 0x1008U);
    EXPECT_EQ(receipt->focus_size, 40U);
    EXPECT_EQ(receipt->route, Route::GpuComposed);
    EXPECT_EQ(receipt->sample_size, SrtSnapshotCompositionReceipt::MaximumSampleBytes);
    EXPECT_EQ(receipt->overlay_count, 2U);
    EXPECT_EQ(receipt->overlay_bytes, 48U);
    EXPECT_EQ(receipt->sample_differing_byte_count, 0U);
    EXPECT_FALSE(receipt->first_differing_focus_offset);
    EXPECT_FALSE(receipt->last_differing_focus_offset);
    for (size_t i = 0; i < receipt->sample_size; ++i) {
        EXPECT_EQ(receipt->cpu_sample[i], cpu[8 + i]);
        EXPECT_EQ(receipt->composed_sample[i], cpu[8 + i]);
    }
}

TEST(SrtSnapshotCompositionReceipt, ReportsOnlyDifferencesInsideTheBoundedFocusSample) {
    std::array<u8, 64> cpu{};
    std::array<u8, 64> composed{};
    for (const size_t focus_offset : {size_t{0}, size_t{7}, size_t{31}, size_t{35}}) {
        composed[8 + focus_offset] = 0x7f;
    }

    const auto receipt = CaptureSrtSnapshotCompositionReceipt(GpuConfiguration, cpu, composed);

    ASSERT_TRUE(receipt);
    EXPECT_EQ(receipt->sample_differing_byte_count, 3U);
    ASSERT_TRUE(receipt->first_differing_focus_offset);
    ASSERT_TRUE(receipt->last_differing_focus_offset);
    EXPECT_EQ(*receipt->first_differing_focus_offset, 0U);
    EXPECT_EQ(*receipt->last_differing_focus_offset, 31U);
}

TEST(SrtSnapshotCompositionReceipt, UsesByteOffsetsForAnUnalignedFocusRange) {
    constexpr SrtSnapshotCompositionReceiptConfig config{
        .snapshot_address = 0x2000,
        .snapshot_size = 16,
        .focus_address = 0x2003,
        .focus_size = 5,
        .route = Route::GpuComposed,
        .overlay_count = 1,
        .overlay_bytes = 4,
    };
    const std::array<u8, 16> cpu{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    auto composed = cpu;
    composed[5] = 0xaa;

    const auto receipt = CaptureSrtSnapshotCompositionReceipt(config, cpu, composed);

    ASSERT_TRUE(receipt);
    EXPECT_EQ(receipt->sample_size, 5U);
    EXPECT_EQ(receipt->cpu_sample[0], 3U);
    EXPECT_EQ(receipt->cpu_sample[4], 7U);
    EXPECT_EQ(receipt->composed_sample[2], 0xaaU);
    EXPECT_EQ(receipt->sample_differing_byte_count, 1U);
    EXPECT_EQ(receipt->first_differing_focus_offset, 2U);
    EXPECT_EQ(receipt->last_differing_focus_offset, 2U);
    EXPECT_TRUE(std::ranges::all_of(std::span{receipt->cpu_sample}.subspan(5),
                                    [](u8 byte) { return byte == 0; }));
    EXPECT_TRUE(std::ranges::all_of(std::span{receipt->composed_sample}.subspan(5),
                                    [](u8 byte) { return byte == 0; }));
}

TEST(SrtSnapshotCompositionReceipt, OwnsSamplesAfterTheInputsAreMutated) {
    std::array<u8, 64> cpu{};
    std::array<u8, 64> composed{};
    cpu[8] = 0x11;
    composed[8] = 0x22;
    const auto receipt = CaptureSrtSnapshotCompositionReceipt(GpuConfiguration, cpu, composed);
    ASSERT_TRUE(receipt);

    cpu.fill(0xcc);
    composed.fill(0xdd);

    EXPECT_EQ(receipt->cpu_sample[0], 0x11U);
    EXPECT_EQ(receipt->composed_sample[0], 0x22U);
    EXPECT_EQ(receipt->sample_differing_byte_count, 1U);
}

TEST(SrtSnapshotCompositionReceipt, CpuOnlyMayMirrorAnAbsentComposedSnapshot) {
    constexpr SrtSnapshotCompositionReceiptConfig config{
        .snapshot_address = 0x3000,
        .snapshot_size = 8,
        .focus_address = 0x3001,
        .focus_size = 4,
        .route = Route::CpuOnly,
    };
    const std::array<u8, 8> cpu{1, 2, 3, 4, 5, 6, 7, 8};

    const auto receipt = CaptureSrtSnapshotCompositionReceipt(config, cpu, {});

    ASSERT_TRUE(receipt);
    EXPECT_EQ(receipt->route, Route::CpuOnly);
    EXPECT_EQ(receipt->sample_size, 4U);
    EXPECT_EQ(receipt->overlay_count, 0U);
    EXPECT_EQ(receipt->overlay_bytes, 0U);
    EXPECT_EQ(receipt->sample_differing_byte_count, 0U);
    EXPECT_TRUE(std::equal(receipt->cpu_sample.begin(), receipt->cpu_sample.begin() + 4,
                           receipt->composed_sample.begin()));
}

TEST(SrtSnapshotCompositionReceipt, UnavailableRouteOwnsOnlyValidatedRangeMetadata) {
    constexpr SrtSnapshotCompositionReceiptConfig config{
        .snapshot_address = 0x4000,
        .snapshot_size = 16,
        .focus_address = 0x4004,
        .focus_size = 4,
        .route = Route::Unavailable,
    };

    const auto receipt = CaptureSrtSnapshotCompositionReceipt(config, {}, {});

    ASSERT_TRUE(receipt);
    EXPECT_EQ(receipt->route, Route::Unavailable);
    EXPECT_EQ(receipt->sample_size, 0U);
    EXPECT_EQ(receipt->sample_differing_byte_count, 0U);
    EXPECT_FALSE(receipt->first_differing_focus_offset);
    EXPECT_FALSE(receipt->last_differing_focus_offset);
}

TEST(SrtSnapshotCompositionReceipt, RejectsOverflowAndOutOfRangeFocus) {
    constexpr auto Last = std::numeric_limits<VAddr>::max();
    const std::array<u8, 16> bytes{};
    for (const auto config : {
             SrtSnapshotCompositionReceiptConfig{Last - 3, 8, Last - 3, 1, Route::CpuOnly},
             SrtSnapshotCompositionReceiptConfig{0x1000, 16, Last - 1, 4, Route::CpuOnly},
             SrtSnapshotCompositionReceiptConfig{0x1000, 16, 0x0fff, 1, Route::CpuOnly},
             SrtSnapshotCompositionReceiptConfig{0x1000, 16, 0x100f, 2, Route::CpuOnly},
             SrtSnapshotCompositionReceiptConfig{0x1000, 0, 0x1000, 1, Route::CpuOnly},
             SrtSnapshotCompositionReceiptConfig{0x1000, 16, 0x1000, 0, Route::CpuOnly},
         }) {
        EXPECT_FALSE(CaptureSrtSnapshotCompositionReceipt(config, bytes, {}));
    }
}

TEST(SrtSnapshotCompositionReceipt, RejectsInconsistentRouteInputsAndOverlayMetadata) {
    const std::array<u8, 64> bytes{};
    auto config = GpuConfiguration;
    EXPECT_FALSE(CaptureSrtSnapshotCompositionReceipt(config, bytes, {}));
    EXPECT_FALSE(CaptureSrtSnapshotCompositionReceipt(config, bytes, std::span{bytes}.first(63)));

    config.route = Route::CpuOnly;
    EXPECT_FALSE(CaptureSrtSnapshotCompositionReceipt(config, bytes, {}));
    config.overlay_count = 0;
    EXPECT_FALSE(CaptureSrtSnapshotCompositionReceipt(config, bytes, {}));
    config.overlay_bytes = 0;
    EXPECT_TRUE(CaptureSrtSnapshotCompositionReceipt(config, bytes, {}));
    EXPECT_FALSE(CaptureSrtSnapshotCompositionReceipt(config, bytes, bytes));

    config.route = Route::Unavailable;
    EXPECT_FALSE(CaptureSrtSnapshotCompositionReceipt(config, bytes, {}));
    EXPECT_TRUE(CaptureSrtSnapshotCompositionReceipt(config, {}, {}));

    config.route = static_cast<Route>(0xff);
    EXPECT_FALSE(CaptureSrtSnapshotCompositionReceipt(config, {}, {}));
}

} // namespace
} // namespace VideoCore
