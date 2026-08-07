// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/device_resident_read_fingerprint.h"
#include "video_core/buffer_cache/diagnostic_readback_pin.h"

using AmdGpu::DeviceResidentReadFingerprintPlanner;
using AmdGpu::DeviceResidentReadObservation;
using AmdGpu::DeviceResidentReadReducer;

namespace {

DeviceResidentReadObservation Observation(u32 source, u64 semantic, u64 offset, u64 size,
                                          u64 serial = 1) {
    return {
        .source_buffer_id = source,
        .semantic_identity = semantic,
        .source_offset = offset,
        .source_buffer_size = 4096,
        .bound_size = size,
        .write_serial = serial,
        .device_local = true,
        .shader_read_only = true,
    };
}

} // namespace

TEST(DeviceResidentReadFingerprint, AcceptsOnlyDepthBackedDeviceLocalReadOnlyDraws) {
    DeviceResidentReadFingerprintPlanner planner;

    planner.BeginDraw();
    planner.Observe(Observation(1, 10, 0, 256));
    planner.CommitDraw(/*has_depth=*/false);

    planner.BeginDraw();
    auto cpu_visible = Observation(2, 20, 0, 256);
    cpu_visible.device_local = false;
    planner.Observe(cpu_visible);
    auto writable = Observation(3, 30, 0, 256);
    writable.shader_read_only = false;
    planner.Observe(writable);
    planner.Observe(Observation(4, 40, 0, 256));
    planner.CommitDraw(/*has_depth=*/true);

    const auto plan = planner.TakeFramePlan();
    EXPECT_EQ(plan.depth_draws, 1);
    EXPECT_EQ(plan.excluded_non_depth_draws, 1);
    EXPECT_EQ(plan.rejected_non_device_local, 1);
    EXPECT_EQ(plan.rejected_writable, 1);
    ASSERT_EQ(plan.range_count, 1);
    EXPECT_EQ(plan.ranges[0].source_buffer_id, 4);
}

TEST(DeviceResidentReadFingerprint, DeduplicatesRepeatedReadsAndPreservesReferences) {
    DeviceResidentReadFingerprintPlanner planner;
    planner.BeginDraw();
    for (u32 i = 0; i < 100; ++i) {
        planner.Observe(Observation(7, 70, 128, 256));
    }
    planner.CommitDraw(/*has_depth=*/true);

    const auto plan = planner.TakeFramePlan();
    ASSERT_EQ(plan.range_count, 1);
    EXPECT_EQ(plan.ranges[0].read_references, 100);
    EXPECT_EQ(plan.observations, 100);
}

TEST(DeviceResidentReadFingerprint, GeneratesCheckedFirstMiddleLastSamples) {
    DeviceResidentReadFingerprintPlanner planner;
    planner.BeginDraw();
    planner.Observe(Observation(1, 10, 128, 256));
    planner.Observe(Observation(2, 20, 512, 64));
    auto out_of_bounds = Observation(3, 30, 4080, 32);
    planner.Observe(out_of_bounds);
    planner.CommitDraw(/*has_depth=*/true);

    const auto plan = planner.TakeFramePlan();
    ASSERT_EQ(plan.range_count, 2);
    EXPECT_EQ(plan.ranges[0].sample_count, 3);
    for (u32 i = 0; i < plan.ranges[0].sample_count; ++i) {
        const auto& sample = plan.ranges[0].samples[i];
        EXPECT_GE(sample.source_offset, 128);
        EXPECT_LE(sample.source_offset + sample.size, 128 + 256);
        EXPECT_EQ(sample.source_offset % 4, 0);
        EXPECT_EQ(sample.size % 4, 0);
    }
    EXPECT_EQ(plan.ranges[1].sample_count, 1);
    EXPECT_EQ(plan.ranges[1].samples[0].size, 64);
    EXPECT_EQ(plan.rejected_out_of_bounds, 1);
}

TEST(DeviceResidentReadFingerprint, BoundsRangesBytesHistoryAndRingWithoutWrap) {
    DeviceResidentReadFingerprintPlanner planner;
    planner.BeginDraw();
    for (u32 i = 0; i < DeviceResidentReadFingerprintPlanner::MaxRangesPerFrame + 1; ++i) {
        planner.Observe(Observation(i + 1, 1000 + i, 0, 256));
    }
    planner.CommitDraw(/*has_depth=*/true);
    const auto plan = planner.TakeFramePlan();
    EXPECT_EQ(plan.range_count, DeviceResidentReadFingerprintPlanner::MaxRangesPerFrame);
    EXPECT_EQ(plan.sample_bytes, DeviceResidentReadFingerprintPlanner::MaxBytesPerFrame);
    EXPECT_EQ(plan.truncated_ranges, 1);

    constexpr u64 atom = 256;
    constexpr u64 padded =
        ((DeviceResidentReadFingerprintPlanner::MaxBytesPerFrame + atom - 1) / atom) * atom;
    EXPECT_EQ(DeviceResidentReadFingerprintPlanner::RequiredWindowBytes(atom),
              DeviceResidentReadFingerprintPlanner::MaxReportFrames * padded);
}

TEST(DeviceResidentReadFingerprint, ReducerReportsChangeAndExactAbaWithoutDigest) {
    DeviceResidentReadReducer reducer;

    const auto first = reducer.Observe(/*identity=*/9, /*fingerprint=*/100);
    EXPECT_TRUE(first.first_observation);
    const auto same = reducer.Observe(9, 100);
    EXPECT_TRUE(same.unchanged);
    const auto changed = reducer.Observe(9, 200);
    EXPECT_TRUE(changed.changed);
    EXPECT_FALSE(changed.exact_aba_return);
    const auto returned = reducer.Observe(9, 100);
    EXPECT_TRUE(returned.changed);
    EXPECT_TRUE(returned.exact_aba_return);
}

TEST(DeviceResidentReadFingerprint, WriteSerialChangeSuppressesStabilityClaim) {
    DeviceResidentReadFingerprintPlanner planner;
    planner.BeginDraw();
    planner.Observe(Observation(1, 10, 0, 256, /*serial=*/4));
    planner.Observe(Observation(1, 10, 0, 256, /*serial=*/5));
    planner.CommitDraw(/*has_depth=*/true);

    const auto plan = planner.TakeFramePlan();
    ASSERT_EQ(plan.range_count, 1);
    EXPECT_TRUE(plan.ranges[0].multi_version);
}

TEST(DeviceResidentReadFingerprint, DiagnosticPinDefersPhysicalDeletion) {
    VideoCore::DiagnosticReadbackPin pin;
    pin.Acquire();
    const auto first_delete = pin.RequestDelete();
    EXPECT_TRUE(first_delete.logical_delete);
    EXPECT_FALSE(first_delete.erase_now);
    EXPECT_TRUE(pin.IsDeletePending());
    EXPECT_TRUE(pin.Release());
    EXPECT_FALSE(pin.IsPinned());

    const auto duplicate_delete = pin.RequestDelete();
    EXPECT_FALSE(duplicate_delete.logical_delete);
    EXPECT_FALSE(duplicate_delete.erase_now);
}
