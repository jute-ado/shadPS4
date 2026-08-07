// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/amdgpu/occlusion_query_reuse_diagnostic.h"

using AmdGpu::OcclusionQueryReuseDiagnostic;

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
