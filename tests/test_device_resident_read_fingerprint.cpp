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
        .gpu_modified = true,
        .shader_read_only = true,
    };
}

} // namespace

TEST(DeviceResidentReadFingerprint, AcceptsOnlyDepthBackedDeviceLocalReadOnlyDraws) {
    DeviceResidentReadFingerprintPlanner planner;

    planner.BeginDraw();
    planner.Observe(Observation(1, 10, 0, 256));
    const auto non_depth_commit = planner.CommitDraw(/*has_depth=*/false);
    EXPECT_EQ(non_depth_commit.source_count, 0);

    planner.BeginDraw();
    auto cpu_visible = Observation(2, 20, 0, 256);
    cpu_visible.device_local = false;
    planner.Observe(cpu_visible);
    auto writable = Observation(3, 30, 0, 256);
    writable.shader_read_only = false;
    planner.Observe(writable);
    auto not_gpu_modified = Observation(5, 50, 0, 256);
    not_gpu_modified.gpu_modified = false;
    planner.Observe(not_gpu_modified);
    planner.Observe(Observation(4, 40, 0, 256));
    const auto committed = planner.CommitDraw(/*has_depth=*/true);

    const auto plan = planner.TakeFramePlan();
    EXPECT_EQ(plan.depth_draws, 1);
    EXPECT_EQ(plan.excluded_non_depth_draws, 1);
    EXPECT_EQ(plan.rejected_non_device_local, 1);
    EXPECT_EQ(plan.rejected_not_gpu_modified, 1);
    EXPECT_EQ(plan.rejected_writable, 1);
    ASSERT_EQ(plan.range_count, 1);
    EXPECT_EQ(plan.ranges[0].source_buffer_id, 4);
    ASSERT_EQ(committed.source_count, 1);
    EXPECT_EQ(committed.source_buffer_ids[0], 4);
}

TEST(DeviceResidentReadFingerprint, DeduplicatesRepeatedReadsAndPreservesReferences) {
    DeviceResidentReadFingerprintPlanner planner;
    planner.BeginDraw();
    for (u32 i = 0; i < 100; ++i) {
        planner.Observe(Observation(7, 70, 128, 256));
    }
    (void)planner.CommitDraw(/*has_depth=*/true);

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
    (void)planner.CommitDraw(/*has_depth=*/true);

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

TEST(DeviceResidentReadFingerprint, SamplesOnlyLogicalBytesAfterDescriptorAlignmentPrefix) {
    DeviceResidentReadFingerprintPlanner planner;
    planner.BeginDraw();
    planner.Observe(Observation(1, 10, /*logical_offset=*/132, /*logical_size=*/256));
    (void)planner.CommitDraw(/*has_depth=*/true);

    const auto plan = planner.TakeFramePlan();
    ASSERT_EQ(plan.range_count, 1);
    ASSERT_EQ(plan.ranges[0].sample_count, 3);
    EXPECT_EQ(plan.ranges[0].samples[0].source_offset, 132);
    for (u32 i = 0; i < plan.ranges[0].sample_count; ++i) {
        const auto& sample = plan.ranges[0].samples[i];
        EXPECT_GE(sample.source_offset, 132);
        EXPECT_LE(sample.source_offset + sample.size, 132 + 256);
    }
}

TEST(DeviceResidentReadFingerprint, BoundsRangesBytesHistoryAndRingWithoutWrap) {
    DeviceResidentReadFingerprintPlanner planner;
    planner.BeginDraw();
    for (u32 i = 0; i < DeviceResidentReadFingerprintPlanner::MaxRangesPerFrame + 1; ++i) {
        planner.Observe(Observation(i + 1, 1000 + i, 0, 256));
    }
    (void)planner.CommitDraw(/*has_depth=*/true);
    const auto plan = planner.TakeFramePlan();
    EXPECT_EQ(plan.range_count, DeviceResidentReadFingerprintPlanner::MaxRangesPerFrame);
    EXPECT_EQ(plan.sample_bytes, DeviceResidentReadFingerprintPlanner::MaxBytesPerFrame);
    EXPECT_EQ(plan.truncated_ranges, 1);

    constexpr u64 atom = 256;
    constexpr u64 padded =
        ((DeviceResidentReadFingerprintPlanner::MaxBytesPerFrame + atom - 1) / atom) * atom;
    EXPECT_EQ(DeviceResidentReadFingerprintPlanner::RequiredWindowBytes(atom),
              DeviceResidentReadFingerprintPlanner::MaxReportFrames * padded);
    EXPECT_GE(DeviceResidentReadFingerprintPlanner::MaxReportFrames, 1700);
    const u64 last_slot = DeviceResidentReadFingerprintPlanner::MaxReportFrames - 1;
    EXPECT_LE(last_slot * padded + padded,
              DeviceResidentReadFingerprintPlanner::RequiredWindowBytes(atom));
}

TEST(DeviceResidentReadFingerprint, BoundsPersistentSemanticHistory) {
    DeviceResidentReadFingerprintPlanner planner;
    for (u32 i = 0; i < DeviceResidentReadFingerprintPlanner::MaxHistoryRanges; ++i) {
        planner.BeginDraw();
        planner.Observe(Observation(1, 1000 + i, 0, 256));
        (void)planner.CommitDraw(/*has_depth=*/true);
        const auto accepted = planner.TakeFramePlan();
        ASSERT_EQ(accepted.range_count, 1);
        ASSERT_EQ(accepted.truncated_history, 0);
    }

    planner.BeginDraw();
    planner.Observe(
        Observation(1, 1000 + DeviceResidentReadFingerprintPlanner::MaxHistoryRanges, 0, 256));
    (void)planner.CommitDraw(/*has_depth=*/true);
    const auto truncated = planner.TakeFramePlan();
    EXPECT_EQ(truncated.range_count, 0);
    EXPECT_EQ(truncated.truncated_history, 1);
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
    (void)planner.CommitDraw(/*has_depth=*/true);

    const auto plan = planner.TakeFramePlan();
    ASSERT_EQ(plan.range_count, 1);
    EXPECT_TRUE(plan.ranges[0].multi_version);
}

TEST(DeviceResidentReadFingerprint, WritableAliasRejectsReadRegardlessOfBindingOrder) {
    DeviceResidentReadFingerprintPlanner planner;
    planner.BeginDraw();
    planner.Observe(Observation(1, 10, 0, 256));
    auto writable = Observation(1, 11, 0, 256);
    writable.shader_read_only = false;
    planner.Observe(writable);
    (void)planner.CommitDraw(/*has_depth=*/true);

    const auto plan = planner.TakeFramePlan();
    EXPECT_EQ(plan.range_count, 0);
    EXPECT_EQ(plan.rejected_writable, 1);
    EXPECT_EQ(plan.rejected_write_alias, 1);

    DeviceResidentReadFingerprintPlanner reverse_planner;
    reverse_planner.BeginDraw();
    reverse_planner.Observe(writable);
    reverse_planner.Observe(Observation(1, 10, 0, 256));
    (void)reverse_planner.CommitDraw(/*has_depth=*/true);
    const auto reverse_plan = reverse_planner.TakeFramePlan();
    EXPECT_EQ(reverse_plan.range_count, 0);
    EXPECT_EQ(reverse_plan.rejected_write_alias, 1);
}

TEST(DeviceResidentReadFingerprint, SequenceGapCannotCreateFalseAba) {
    DeviceResidentReadReducer reducer;
    EXPECT_TRUE(reducer.Observe(9, 100, /*sequence=*/10).first_observation);
    EXPECT_TRUE(reducer.Observe(9, 200, /*sequence=*/11).changed);
    const auto after_busy_frame = reducer.Observe(9, 100, /*sequence=*/13);
    EXPECT_FALSE(after_busy_frame.changed);
    EXPECT_FALSE(after_busy_frame.exact_aba_return);
}

TEST(DeviceResidentReadFingerprint, CollectsOnlyInsideRequestedFrameWindow) {
    EXPECT_FALSE(DeviceResidentReadFingerprintPlanner::ShouldCollect(/*sequence=*/3399,
                                                                     /*start=*/3400,
                                                                     /*end=*/3912));
    EXPECT_TRUE(DeviceResidentReadFingerprintPlanner::ShouldCollect(/*sequence=*/3400,
                                                                    /*start=*/3400,
                                                                    /*end=*/3912));
    EXPECT_TRUE(DeviceResidentReadFingerprintPlanner::ShouldCollect(/*sequence=*/3911,
                                                                    /*start=*/3400,
                                                                    /*end=*/3912));
    EXPECT_FALSE(DeviceResidentReadFingerprintPlanner::ShouldCollect(/*sequence=*/3912,
                                                                     /*start=*/3400,
                                                                     /*end=*/3912));
    EXPECT_FALSE(DeviceResidentReadFingerprintPlanner::ShouldObserveDraw(/*draw=*/233,
                                                                         /*minimum_draw=*/234));
    EXPECT_TRUE(DeviceResidentReadFingerprintPlanner::ShouldObserveDraw(/*draw=*/234,
                                                                        /*minimum_draw=*/234));
}

TEST(DeviceResidentReadFingerprint, SemanticOrdinalSurvivesPhysicalDeduplication) {
    DeviceResidentReadFingerprintPlanner planner;
    constexpr u64 first_semantic = (u64{77} << 32) | (u64{3} << 24) | 11;
    constexpr u64 second_semantic = (u64{88} << 32) | (u64{4} << 24) | 12;
    planner.BeginDraw();
    planner.Observe(Observation(1, first_semantic, 128, 256));
    (void)planner.CommitDraw(/*has_depth=*/true);
    planner.BeginDraw();
    planner.Observe(Observation(1, second_semantic, 128, 256));
    (void)planner.CommitDraw(/*has_depth=*/true);

    const auto plan = planner.TakeFramePlan();
    ASSERT_EQ(plan.range_count, 1);
    EXPECT_EQ(plan.ranges[0].semantic_identity, first_semantic);
    EXPECT_EQ(plan.ranges[0].read_references, 2);
    const auto decoded = DeviceResidentReadFingerprintPlanner::DecodeSemanticIdentity(
        plan.ranges[0].semantic_identity);
    EXPECT_EQ(decoded.draw, 77);
    EXPECT_EQ(decoded.stage, 3);
    EXPECT_EQ(decoded.binding, 11);
}

TEST(DeviceResidentReadFingerprint, PinFailureMarksEveryRangeForSourceUnavailable) {
    DeviceResidentReadFingerprintPlanner planner;
    planner.BeginDraw();
    planner.Observe(Observation(7, 10, 0, 256));
    planner.Observe(Observation(7, 11, 512, 256));
    (void)planner.CommitDraw(/*has_depth=*/true);
    planner.RecordPinFailure(7);

    const auto plan = planner.TakeFramePlan();
    ASSERT_EQ(plan.range_count, 2);
    EXPECT_EQ(plan.pin_failures, 1);
    EXPECT_TRUE(plan.ranges[0].source_unavailable);
    EXPECT_TRUE(plan.ranges[1].source_unavailable);
}

TEST(DeviceResidentReadFingerprint, WriteGenerationAdvancesForEveryWaw) {
    VideoCore::DiagnosticWriteGeneration generation;
    EXPECT_EQ(generation.Serial(), 0);
    generation.MarkWrite();
    generation.MarkWrite();
    EXPECT_EQ(generation.Serial(), 2);
}

TEST(DeviceResidentReadFingerprint, DiagnosticPinDefersPhysicalDeletion) {
    VideoCore::DiagnosticReadbackPin pin;
    pin.Acquire();
    pin.Acquire();
    const auto first_delete = pin.RequestDelete();
    EXPECT_TRUE(first_delete.logical_delete);
    EXPECT_FALSE(first_delete.erase_now);
    EXPECT_TRUE(pin.IsDeletePending());
    EXPECT_FALSE(pin.Release());
    EXPECT_TRUE(pin.IsPinned());
    EXPECT_TRUE(pin.Release());
    EXPECT_FALSE(pin.IsPinned());

    const auto duplicate_delete = pin.RequestDelete();
    EXPECT_FALSE(duplicate_delete.logical_delete);
    EXPECT_FALSE(duplicate_delete.erase_now);
}
