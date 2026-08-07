// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/amdgpu/occlusion_query_reuse_diagnostic.h"
#include "video_core/amdgpu/pixel_pipe_stat_control.h"

using AmdGpu::OcclusionQueryReuseDiagnostic;
using AmdGpu::PixelPipeStatControl;

TEST(OcclusionQueryReuseDiagnostic, ReportsFreshReuseAndPriorValidityWithoutAddresses) {
    OcclusionQueryReuseDiagnostic diagnostic{/*report_interval=*/2, /*report_limit=*/2,
                                             /*target_limit=*/4};
    std::array<u64, 4> results{};

    EXPECT_FALSE(diagnostic.Observe(0x1000, results.data(), 2).has_value());

    results[0] = OcclusionQueryReuseDiagnostic::ValidMask | 11;
    results[2] = OcclusionQueryReuseDiagnostic::ValidMask | 12;
    const auto first = diagnostic.Observe(0x1000, results.data(), 2);

    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->sequence, 1);
    EXPECT_EQ(first->dumps, 2);
    EXPECT_EQ(first->fresh_targets, 1);
    EXPECT_EQ(first->reused_targets, 1);
    EXPECT_EQ(first->unknown_targets, 0);
    EXPECT_EQ(first->no_prior_valid, 1);
    EXPECT_EQ(first->partial_prior_valid, 0);
    EXPECT_EQ(first->all_prior_valid, 1);
    EXPECT_EQ(first->distinct_targets, 1);
    EXPECT_EQ(first->min_counter_pairs, 2);
    EXPECT_EQ(first->max_counter_pairs, 2);

    results[2] = 12;
    EXPECT_FALSE(diagnostic.Observe(0x2000, results.data(), 2).has_value());
    results[0] = 11;
    const auto second = diagnostic.Observe(0x3000, results.data(), 2);

    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->fresh_targets, 2);
    EXPECT_EQ(second->partial_prior_valid, 1);
    EXPECT_EQ(second->no_prior_valid, 1);
    EXPECT_EQ(second->distinct_targets, 3);

    EXPECT_FALSE(diagnostic.Observe(0x1000, results.data(), 2).has_value());
    EXPECT_FALSE(diagnostic.Observe(0x1000, results.data(), 2).has_value());
}

TEST(OcclusionQueryReuseDiagnostic, BoundsRememberedTargets) {
    OcclusionQueryReuseDiagnostic diagnostic{/*report_interval=*/3, /*report_limit=*/1,
                                             /*target_limit=*/1};
    std::array<u64, 2> results{};

    EXPECT_FALSE(diagnostic.Observe(0x1000, results.data(), 1).has_value());
    EXPECT_FALSE(diagnostic.Observe(0x2000, results.data(), 1).has_value());
    const auto snapshot = diagnostic.Observe(0x1000, results.data(), 1);

    ASSERT_TRUE(snapshot.has_value());
    EXPECT_EQ(snapshot->fresh_targets, 1);
    EXPECT_EQ(snapshot->unknown_targets, 1);
    EXPECT_EQ(snapshot->reused_targets, 1);
    EXPECT_EQ(snapshot->distinct_targets, 1);
}

TEST(OcclusionQueryReuseDiagnostic, ReportsActiveControlLayoutOnDumps) {
    OcclusionQueryReuseDiagnostic diagnostic{/*report_interval=*/2, /*report_limit=*/2,
                                             /*target_limit=*/4};
    std::array<u64, 8> results{};
    diagnostic.ObserveControl(PixelPipeStatControl{
        .counter_id = 7, .stride_bytes = 8, .instance_enable_mask = 0b0101});

    EXPECT_FALSE(diagnostic.Observe(0x1000, results.data(), 4).has_value());
    const auto mismatched = diagnostic.Observe(0x1000, results.data(), 4);

    ASSERT_TRUE(mismatched.has_value());
    EXPECT_EQ(mismatched->controls, 1);
    EXPECT_EQ(mismatched->control_changes, 0);
    EXPECT_EQ(mismatched->dumps_without_control, 0);
    EXPECT_EQ(mismatched->hardcoded_layout_mismatches, 2);
    EXPECT_EQ(mismatched->counter_id_min, 7);
    EXPECT_EQ(mismatched->counter_id_max, 7);
    EXPECT_EQ(mismatched->stride_bytes_min, 8);
    EXPECT_EQ(mismatched->stride_bytes_max, 8);
    EXPECT_EQ(mismatched->enabled_instances_min, 2);
    EXPECT_EQ(mismatched->enabled_instances_max, 2);
    EXPECT_EQ(mismatched->instance_mask_and, 0b0101);
    EXPECT_EQ(mismatched->instance_mask_or, 0b0101);

    diagnostic.ObserveControl(PixelPipeStatControl{
        .counter_id = 0, .stride_bytes = 16, .instance_enable_mask = 0b1111});
    EXPECT_FALSE(diagnostic.Observe(0x1000, results.data(), 4).has_value());
    const auto matching = diagnostic.Observe(0x1000, results.data(), 4);

    ASSERT_TRUE(matching.has_value());
    EXPECT_EQ(matching->controls, 1);
    EXPECT_EQ(matching->control_changes, 1);
    EXPECT_EQ(matching->hardcoded_layout_mismatches, 0);
    EXPECT_EQ(matching->counter_id_min, 0);
    EXPECT_EQ(matching->stride_bytes_min, 16);
    EXPECT_EQ(matching->enabled_instances_min, 4);
    EXPECT_EQ(matching->instance_mask_and, 0b1111);
}
