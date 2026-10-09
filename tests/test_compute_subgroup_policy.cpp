// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/compute_subgroup_policy.h"

TEST(ComputeSubgroupPolicy, DoesNotForceGuestSizeWithoutSubgroupOperations) {
    EXPECT_FALSE(Vulkan::ShouldRequireGuestSubgroupSize(true, false));
}

TEST(ComputeSubgroupPolicy, ForcesGuestSizeForSubgroupOperations) {
    EXPECT_TRUE(Vulkan::ShouldRequireGuestSubgroupSize(true, true));
}

TEST(ComputeSubgroupPolicy, CannotForceUnsupportedGuestSize) {
    EXPECT_FALSE(Vulkan::ShouldRequireGuestSubgroupSize(false, true));
}
