// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_diagnostic_checkpoint.h"

using Vulkan::DiagnosticCheckpointKind;
using Vulkan::DiagnosticCheckpointPhase;
using Vulkan::MakeDiagnosticCheckpoint;

TEST(DiagnosticCheckpoint, DistinguishesBeforeAndAfterMarkersForOneDispatch) {
    constexpr std::array<u32, 3> dimensions{180, 1, 1};
    const auto* before = MakeDiagnosticCheckpoint(DiagnosticCheckpointKind::Compute, 0x4defcf7f,
                                                  0, DiagnosticCheckpointPhase::Before,
                                                  dimensions);
    const auto* after = MakeDiagnosticCheckpoint(DiagnosticCheckpointKind::Compute, 0x4defcf7f, 0,
                                                 DiagnosticCheckpointPhase::After, dimensions);

    ASSERT_NE(before, nullptr);
    ASSERT_NE(after, nullptr);
    EXPECT_NE(before, after);
    EXPECT_EQ(before->phase, DiagnosticCheckpointPhase::Before);
    EXPECT_EQ(after->phase, DiagnosticCheckpointPhase::After);
    EXPECT_EQ(before->runtime, dimensions);
    EXPECT_EQ(after->runtime, dimensions);
}

TEST(DiagnosticCheckpoint, ReusesStableStorageForAnIdenticalMarker) {
    constexpr std::array<u32, 3> dimensions{7, 7, 1};
    const auto* first = MakeDiagnosticCheckpoint(DiagnosticCheckpointKind::Compute, 0x87cf9b8a, 0,
                                                 DiagnosticCheckpointPhase::Before, dimensions);
    const auto* second = MakeDiagnosticCheckpoint(DiagnosticCheckpointKind::Compute, 0x87cf9b8a, 0,
                                                  DiagnosticCheckpointPhase::Before, dimensions);

    EXPECT_EQ(first, second);
}
