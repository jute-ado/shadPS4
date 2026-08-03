// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/pipeline_bind_tracker.h"

TEST(PipelineBindTracker, SuppressesOnlyRedundantBindsWithinOneCommandBuffer) {
    Vulkan::PipelineBindTracker<unsigned> tracker;

    EXPECT_TRUE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Graphics, 11));
    EXPECT_FALSE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Graphics, 11));
    EXPECT_TRUE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Graphics, 12));
    EXPECT_FALSE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Graphics, 12));

    EXPECT_TRUE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Compute, 12));
    EXPECT_FALSE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Compute, 12));

    tracker.Reset();

    EXPECT_TRUE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Graphics, 12));
    EXPECT_TRUE(tracker.NeedsBind(Vulkan::PipelineBindPoint::Compute, 12));
}
