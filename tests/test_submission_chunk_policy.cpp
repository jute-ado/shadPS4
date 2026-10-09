// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_submission_chunk_policy.h"

using Vulkan::SubmissionChunkPolicy;
using Vulkan::SubmissionBackpressureWaitTick;

TEST(SubmissionChunkPolicy, SplitsExactlyAtConfiguredWorkBudget) {
    SubmissionChunkPolicy policy{3};
    EXPECT_FALSE(policy.RecordGpuWork());
    EXPECT_FALSE(policy.RecordGpuWork());
    EXPECT_TRUE(policy.RecordGpuWork());
}

TEST(SubmissionChunkPolicy, StartsANewChunkAfterSplit) {
    SubmissionChunkPolicy policy{2};
    EXPECT_FALSE(policy.RecordGpuWork());
    EXPECT_TRUE(policy.RecordGpuWork());
    EXPECT_FALSE(policy.RecordGpuWork());
    EXPECT_TRUE(policy.RecordGpuWork());
}

TEST(SubmissionChunkPolicy, ZeroBudgetDisablesAutomaticSplitting) {
    SubmissionChunkPolicy policy{0};
    for (int i = 0; i < 1000; ++i) {
        EXPECT_FALSE(policy.RecordGpuWork());
    }
}

TEST(SubmissionBackpressure, DoesNotWaitInsideLeadBudget) {
    EXPECT_FALSE(SubmissionBackpressureWaitTick(8, 0, 8).has_value());
    EXPECT_FALSE(SubmissionBackpressureWaitTick(12, 8, 8).has_value());
}

TEST(SubmissionBackpressure, WaitsForOldestExcessSubmission) {
    EXPECT_EQ(SubmissionBackpressureWaitTick(9, 0, 8), 1);
    EXPECT_EQ(SubmissionBackpressureWaitTick(20, 5, 8), 12);
}
