// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/diagnostic_gpu_marker_policy.h"

namespace Vulkan {
namespace {

TEST(DiagnosticGpuMarkerPolicy, RequiresExplicitDiagnosticOptIn) {
    EXPECT_FALSE(ShouldEnableDiagnosticGpuMarkers(true, false));
    EXPECT_FALSE(ShouldEnableDiagnosticGpuMarkers(false, true));
    EXPECT_TRUE(ShouldEnableDiagnosticGpuMarkers(true, true));
}

} // namespace
} // namespace Vulkan
