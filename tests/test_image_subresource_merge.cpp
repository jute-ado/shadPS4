// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/texture_cache/image_subresource_merge.h"

namespace VideoCore {

TEST(ImageSubresourceMerge, PreservesRenderTargetContentsInAvailableParent) {
    const auto plan = PlanImageSubresourceMerge(false, true, true);

    EXPECT_TRUE(plan.copy_contents);
    EXPECT_TRUE(plan.rebind_source);
    EXPECT_TRUE(plan.propagate_target);
    EXPECT_TRUE(plan.release_source);
}

TEST(ImageSubresourceMerge, RetainsRenderTargetUntilParentExists) {
    const auto plan = PlanImageSubresourceMerge(false, true, false);

    EXPECT_FALSE(plan.copy_contents);
    EXPECT_FALSE(plan.release_source);
}

TEST(ImageSubresourceMerge, CopiesOrdinarySubresourceIntoAvailableParent) {
    const auto plan = PlanImageSubresourceMerge(false, false, true);

    EXPECT_TRUE(plan.copy_contents);
    EXPECT_FALSE(plan.propagate_target);
    EXPECT_TRUE(plan.release_source);
}

TEST(ImageSubresourceMerge, RebindsBoundOrdinarySubresourceBeforeRelease) {
    const auto plan = PlanImageSubresourceMerge(true, false, true);

    EXPECT_TRUE(plan.copy_contents);
    EXPECT_TRUE(plan.rebind_source);
    EXPECT_TRUE(plan.release_source);
}

} // namespace VideoCore
