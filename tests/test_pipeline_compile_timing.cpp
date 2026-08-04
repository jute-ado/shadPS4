// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/pipeline_compile_timing.h"

TEST(PipelineCompileTiming, RequiresExactOptInValue) {
    EXPECT_FALSE(Vulkan::PipelineCompileTimingRequested(nullptr));
    EXPECT_FALSE(Vulkan::PipelineCompileTimingRequested(""));
    EXPECT_TRUE(Vulkan::PipelineCompileTimingRequested("1"));
    EXPECT_FALSE(Vulkan::PipelineCompileTimingRequested("0"));
    EXPECT_FALSE(Vulkan::PipelineCompileTimingRequested("true"));
    EXPECT_FALSE(Vulkan::PipelineCompileTimingRequested("10"));
}

TEST(PipelineCompileTiming, ReportsOnlyIndividuallySlowCompiles) {
    EXPECT_FALSE(Vulkan::PipelineCompileTimingShouldReport(0));
    EXPECT_FALSE(Vulkan::PipelineCompileTimingShouldReport(4'999'999));
    EXPECT_TRUE(Vulkan::PipelineCompileTimingShouldReport(5'000'000));
}
