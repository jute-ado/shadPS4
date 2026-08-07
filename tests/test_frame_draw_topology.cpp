// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/frame_draw_topology_diagnostic.h"

namespace VideoCore {
namespace {

TEST(FrameDrawTopologyDiagnostic, SnapshotsAndClearsBoundedFrameCounters) {
    FrameDrawTopologyDiagnostic tracker{2};

    tracker.ObserveDraw(DrawTopologyKind::Direct);
    tracker.ObserveDraw(DrawTopologyKind::DirectIndexed);
    tracker.ObserveDraw(DrawTopologyKind::Indirect);
    tracker.ObserveDraw(DrawTopologyKind::IndirectIndexed);
    tracker.ObserveDrawResult(DrawTopologyResult::Submitted);
    tracker.ObserveDrawResult(DrawTopologyResult::Filtered);
    tracker.ObserveDrawResult(DrawTopologyResult::MissingPipeline);
    tracker.ObserveDrawResult(DrawTopologyResult::BindingFailed);
    tracker.ObserveOcclusion(OcclusionEventKind::Control);
    tracker.ObserveOcclusion(OcclusionEventKind::Dump);
    tracker.ObserveOcclusion(OcclusionEventKind::Reset);

    const auto first = tracker.TakeSnapshot();
    EXPECT_EQ(first.sequence, 1);
    EXPECT_TRUE(first.should_report);
    EXPECT_EQ(first.direct, 1);
    EXPECT_EQ(first.direct_indexed, 1);
    EXPECT_EQ(first.indirect, 1);
    EXPECT_EQ(first.indirect_indexed, 1);
    EXPECT_EQ(first.submitted, 1);
    EXPECT_EQ(first.filtered, 1);
    EXPECT_EQ(first.missing_pipeline, 1);
    EXPECT_EQ(first.binding_failed, 1);
    EXPECT_EQ(first.occlusion_control, 1);
    EXPECT_EQ(first.occlusion_dump, 1);
    EXPECT_EQ(first.occlusion_reset, 1);

    const auto second = tracker.TakeSnapshot();
    EXPECT_EQ(second.sequence, 2);
    EXPECT_TRUE(second.should_report);
    EXPECT_EQ(second.direct + second.direct_indexed + second.indirect + second.indirect_indexed, 0);
    EXPECT_EQ(second.submitted + second.filtered + second.missing_pipeline +
                  second.binding_failed,
              0);

    const auto third = tracker.TakeSnapshot();
    EXPECT_EQ(third.sequence, 3);
    EXPECT_FALSE(third.should_report);
}

} // namespace
} // namespace VideoCore
