// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/frame_draw_topology_diagnostic.h"

namespace VideoCore {
namespace {

TEST(FrameDrawTopologyDiagnostic, SnapshotsAndClearsBoundedFrameCounters) {
    FrameDrawTopologyDiagnostic tracker{2};

    tracker.ObservePacket(DrawTopologyPacket::DrawIndex2);
    tracker.ObservePacket(DrawTopologyPacket::DrawIndirect);
    tracker.ObservePacket(DrawTopologyPacket::SetBase);
    tracker.ObservePacket(DrawTopologyPacket::IndexBufferSize);
    tracker.ObservePacket(DrawTopologyPacket::SetPredication);
    tracker.ObserveDraw(DrawTopologyKind::Direct);
    tracker.ObserveDraw(DrawTopologyKind::DirectIndexed);
    tracker.ObserveDraw(DrawTopologyKind::Indirect);
    tracker.ObserveDraw(DrawTopologyKind::IndirectIndexed);
    tracker.ObserveDrawResult(DrawTopologyResult::Submitted);
    tracker.ObserveDrawResult(DrawTopologyResult::FilterFastClear);
    tracker.ObserveDrawResult(DrawTopologyResult::FilterFmaskDecompress);
    tracker.ObserveDrawResult(DrawTopologyResult::FilterResolve);
    tracker.ObserveDrawResult(DrawTopologyResult::FilterPrimitiveNone);
    tracker.ObserveDrawResult(DrawTopologyResult::FilterDepthStencilCopy);
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
    EXPECT_EQ(first.filtered, 5);
    EXPECT_EQ(first.filter_fast_clear, 1);
    EXPECT_EQ(first.filter_fmask_decompress, 1);
    EXPECT_EQ(first.filter_resolve, 1);
    EXPECT_EQ(first.filter_primitive_none, 1);
    EXPECT_EQ(first.filter_depth_stencil_copy, 1);
    EXPECT_EQ(first.missing_pipeline, 1);
    EXPECT_EQ(first.binding_failed, 1);
    EXPECT_EQ(first.occlusion_control, 1);
    EXPECT_EQ(first.occlusion_dump, 1);
    EXPECT_EQ(first.occlusion_reset, 1);
    EXPECT_EQ(first.set_base, 1);
    EXPECT_EQ(first.index_buffer_size, 1);
    EXPECT_EQ(first.set_predication, 1);
    EXPECT_NE(first.packet_hash, FrameDrawTopologyDiagnostic::EmptyPacketHash());

    const auto second = tracker.TakeSnapshot();
    EXPECT_EQ(second.sequence, 2);
    EXPECT_TRUE(second.should_report);
    EXPECT_EQ(second.direct + second.direct_indexed + second.indirect + second.indirect_indexed, 0);
    EXPECT_EQ(second.submitted + second.filtered + second.missing_pipeline +
                  second.binding_failed,
              0);
    EXPECT_EQ(second.packet_hash, FrameDrawTopologyDiagnostic::EmptyPacketHash());

    const auto third = tracker.TakeSnapshot();
    EXPECT_EQ(third.sequence, 3);
    EXPECT_FALSE(third.should_report);
}

TEST(FrameDrawTopologyDiagnostic, OrderedPacketFingerprintChangesWithTopology) {
    FrameDrawTopologyDiagnostic first_tracker{1};
    FrameDrawTopologyDiagnostic same_tracker{1};
    FrameDrawTopologyDiagnostic changed_tracker{1};

    for (auto* tracker : {&first_tracker, &same_tracker}) {
        tracker->ObservePacket(DrawTopologyPacket::DrawIndex2);
        tracker->ObservePacket(DrawTopologyPacket::SetBase);
        tracker->ObservePacket(DrawTopologyPacket::DrawIndirect);
    }
    changed_tracker.ObservePacket(DrawTopologyPacket::DrawIndex2);
    changed_tracker.ObservePacket(DrawTopologyPacket::IndexBufferSize);
    changed_tracker.ObservePacket(DrawTopologyPacket::DrawIndirect);

    const auto first = first_tracker.TakeSnapshot();
    const auto same = same_tracker.TakeSnapshot();
    const auto changed = changed_tracker.TakeSnapshot();
    EXPECT_EQ(first.packet_hash, same.packet_hash);
    EXPECT_NE(first.packet_hash, changed.packet_hash);
}

} // namespace
} // namespace VideoCore
