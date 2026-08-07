// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/amdgpu/indirect_argument_readback_plan.h"

namespace AmdGpu {
namespace {

TEST(IndirectArgumentReadbackPlan, PacksRecordsAndPreservesStableIdentityAcrossReordering) {
    IndirectArgumentReadbackPlanner planner;

    for (u32 i = 0; i < 9; ++i) {
        planner.Observe({
            .source_token = i < 5 ? 10U : 20U,
            .range_identity = 0x1000U + i * 20U,
            .source_offset = i * 20U,
            .stride = IndirectArgumentReadbackPlanner::CommandBytes,
            .max_count = 1,
            .indexed = true,
            .gpu_modified = true,
        });
    }

    const auto first = planner.TakeFramePlan();
    ASSERT_EQ(first.record_count, 9U);
    EXPECT_EQ(first.batch_count, 2U);
    for (u32 i = 0; i < first.record_count; ++i) {
        EXPECT_EQ(first.records[i].destination_offset, i * 20U);
        EXPECT_EQ(first.records[i].stable_identity, i);
    }

    for (s32 i = 8; i >= 0; --i) {
        planner.Observe({
            .source_token = i < 5 ? 10U : 20U,
            .range_identity = 0x1000U + static_cast<u32>(i) * 20U,
            .source_offset = static_cast<u32>(i) * 20U,
            .stride = IndirectArgumentReadbackPlanner::CommandBytes,
            .max_count = 1,
            .indexed = true,
            .gpu_modified = true,
        });
    }

    const auto reordered = planner.TakeFramePlan();
    ASSERT_EQ(reordered.record_count, 9U);
    EXPECT_EQ(reordered.records[0].stable_identity, 8U);
    EXPECT_EQ(reordered.records[8].stable_identity, 0U);
}

TEST(IndirectArgumentReadbackPlan, RejectsUnsupportedAndBoundsRecordCount) {
    IndirectArgumentReadbackPlanner planner;
    auto observation = IndirectArgumentReadbackObservation{
        .source_token = 1,
        .range_identity = 0x2000,
        .source_offset = 0,
        .stride = IndirectArgumentReadbackPlanner::CommandBytes,
        .max_count = 1,
        .indexed = true,
        .gpu_modified = true,
    };

    auto unsupported = observation;
    unsupported.gpu_modified = false;
    planner.Observe(unsupported);
    unsupported = observation;
    unsupported.indexed = false;
    planner.Observe(unsupported);
    unsupported = observation;
    unsupported.stride = 16;
    planner.Observe(unsupported);
    unsupported = observation;
    unsupported.source_offset = std::numeric_limits<u64>::max() - 10;
    planner.Observe(unsupported);

    for (u32 i = 0; i < IndirectArgumentReadbackPlanner::MaxRecordsPerFrame + 1; ++i) {
        observation.range_identity += 0x100;
        observation.source_offset = i * IndirectArgumentReadbackPlanner::CommandBytes;
        planner.Observe(observation);
    }

    const auto plan = planner.TakeFramePlan();
    EXPECT_EQ(plan.record_count, IndirectArgumentReadbackPlanner::MaxRecordsPerFrame);
    EXPECT_EQ(plan.rejected_cpu_visible, 1U);
    EXPECT_EQ(plan.rejected_nonindexed, 1U);
    EXPECT_EQ(plan.rejected_stride, 1U);
    EXPECT_EQ(plan.rejected_overflow, 1U);
    EXPECT_EQ(plan.truncated_records, 1U);
}

TEST(IndirectArgumentReadbackPlan, RingReservationNeverWaitsOrWraps) {
    static_assert(IndirectArgumentReadbackPlanner::RequiredWindowBytes <=
                  IndirectArgumentReadbackPlanner::DownloadRingBytes);

    const auto reservation = TryReserveIndirectReadback(100, 9 * 20, false);
    ASSERT_TRUE(reservation.has_value());
    EXPECT_EQ(reservation->offset, 100U);
    EXPECT_EQ(reservation->next_offset, 280U);

    EXPECT_FALSE(TryReserveIndirectReadback(100, 9 * 20, true).has_value());
    EXPECT_FALSE(TryReserveIndirectReadback(
                     IndirectArgumentReadbackPlanner::DownloadRingBytes - 100, 180, false)
                     .has_value());
}

TEST(IndirectArgumentReadbackPlan, ReducesChangesAndZeroCountsWithoutExposingValues) {
    IndirectArgumentReadbackReducer reducer;
    constexpr std::array<u32, 5> First{3, 1, 2, 4, 5};
    constexpr std::array<u32, 5> Changed{0, 0, 2, 4, 9};

    auto report = reducer.Observe(7, First);
    EXPECT_TRUE(report.first_observation);
    EXPECT_EQ(report.changed_field_mask, 0U);
    EXPECT_FALSE(report.zero_index_count);
    EXPECT_FALSE(report.zero_instance_count);

    report = reducer.Observe(7, First);
    EXPECT_FALSE(report.first_observation);
    EXPECT_EQ(report.changed_field_mask, 0U);

    report = reducer.Observe(7, Changed);
    EXPECT_EQ(report.changed_field_mask, (1U << 0) | (1U << 1) | (1U << 4));
    EXPECT_TRUE(report.zero_index_count);
    EXPECT_TRUE(report.zero_instance_count);
}

TEST(IndirectArgumentReadbackPlan, DefersDeletionOnlyUntilRecordedCopyReleasesPin) {
    DiagnosticReadbackPin pin;

    pin.Acquire();
    EXPECT_FALSE(pin.RequestDelete());
    EXPECT_TRUE(pin.IsDeletePending());
    EXPECT_TRUE(pin.Release());
    EXPECT_FALSE(pin.IsPinned());

    EXPECT_TRUE(pin.RequestDelete());
}

} // namespace
} // namespace AmdGpu
