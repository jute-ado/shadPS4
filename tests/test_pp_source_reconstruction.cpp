// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/host_passes/pp_source_reconstruction.h"

namespace Vulkan::HostPasses {
namespace {

constexpr u64 SlotBytes = 16u << 20;
constexpr u64 ExistingPairedBytes = 12'107'776;

[[nodiscard]] constexpr PpSourceReconstructionDescriptor RouteDescriptor() {
    return {
        .enabled = true,
        .in_window = true,
        .frame_is_new = true,
        .visible_pp_draw_encoded = true,
        .sampled_metadata_valid = true,
        .fsr_bypassed = true,
        .source_view_matches_baseline = true,
        .source_view_srgb = true,
        .source_snapshot_available = true,
        .source_snapshot_view_available = true,
        .reconstruction_output_available = true,
        .source_width = 1920,
        .source_height = 1080,
        .output_width = 1280,
        .output_height = 720,
        .source_format = FinalGuestSurfaceFormat::Bgra8,
        .output_format = FinalGuestSurfaceFormat::Bgra8,
        .existing_readback_bytes = ExistingPairedBytes,
        .slot_bytes = SlotBytes,
        .alignment = 16,
    };
}

TEST(PpSourceReconstruction, DisabledOutsideOrReusedDoesNoWork) {
    auto descriptor = RouteDescriptor();
    descriptor.enabled = false;
    auto plan = PlanPpSourceReconstruction(descriptor);
    EXPECT_EQ(plan.status, FinalGuestSurfaceStatus::AlreadyConsumed);
    EXPECT_EQ(plan.resource_image_count, 0u);
    EXPECT_EQ(plan.total_pp_draw_count, 0u);
    EXPECT_EQ(plan.source_copy_count, 0u);
    EXPECT_EQ(plan.present_copy_count, 0u);

    descriptor = RouteDescriptor();
    descriptor.in_window = false;
    EXPECT_EQ(PlanPpSourceReconstruction(descriptor).total_pp_draw_count, 0u);
    descriptor = RouteDescriptor();
    descriptor.frame_is_new = false;
    EXPECT_EQ(PlanPpSourceReconstruction(descriptor).total_pp_draw_count, 0u);
}

TEST(PpSourceReconstruction, ReplaysImmutableFullSourceAfterVisibleDraw) {
    const auto plan = PlanPpSourceReconstruction(RouteDescriptor());
    EXPECT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(plan.resource_image_count, 2u);
    EXPECT_EQ(plan.source_snapshot_bytes, 1920u * 1080u * 4u);
    EXPECT_EQ(plan.reconstruction_output_bytes, 1280u * 720u * 4u);
    EXPECT_EQ(plan.source_copy_count, 1u);
    EXPECT_EQ(plan.reconstruction_draw_count, 1u);
    EXPECT_EQ(plan.total_pp_draw_count, 2u);
    EXPECT_TRUE(plan.visible_draw_precedes_source_snapshot);
    EXPECT_TRUE(plan.source_snapshot_precedes_reconstruction_draw);
    EXPECT_TRUE(plan.snapshot_is_immutable_during_reconstruction);
    EXPECT_TRUE(plan.same_pp_pipeline);
    EXPECT_TRUE(plan.same_pp_sampler);
    EXPECT_TRUE(plan.same_pp_settings);
    EXPECT_TRUE(plan.same_source_view_format_and_swizzle);
}

TEST(PpSourceReconstruction, FitsExistingFourPlaneReadbackSlot) {
    const auto plan = PlanPpSourceReconstruction(RouteDescriptor());
    EXPECT_EQ(plan.reconstruction_readback_offset, ExistingPairedBytes);
    EXPECT_EQ(plan.final_readback_bytes, 15'794'176u);
    EXPECT_LE(plan.final_readback_bytes, SlotBytes);
    EXPECT_EQ(plan.present_copy_count, 1u);
    EXPECT_EQ(plan.total_present_plane_count, 4u);
}

TEST(PpSourceReconstruction, NeverWaitsOrRetainsGpuObjectsInCallback) {
    const auto plan = PlanPpSourceReconstruction(RouteDescriptor());
    EXPECT_FALSE(plan.cpu_wait);
    EXPECT_FALSE(plan.finish);
    EXPECT_FALSE(plan.callback_retains_frame);
    EXPECT_FALSE(plan.callback_retains_source_image);
    EXPECT_FALSE(plan.callback_retains_reconstruction_image);
    EXPECT_TRUE(plan.callback_payload_is_scalar_only);
}

TEST(PpSourceReconstruction, BaselineMetadataAndExactFormatsFailClosed) {
    auto descriptor = RouteDescriptor();
    descriptor.sampled_metadata_valid = false;
    EXPECT_EQ(PlanPpSourceReconstruction(descriptor).status,
              FinalGuestSurfaceStatus::InvalidationLoss);
    descriptor = RouteDescriptor();
    descriptor.source_view_matches_baseline = false;
    EXPECT_EQ(PlanPpSourceReconstruction(descriptor).status,
              FinalGuestSurfaceStatus::InvalidationLoss);
    descriptor = RouteDescriptor();
    descriptor.source_view_srgb = false;
    EXPECT_EQ(PlanPpSourceReconstruction(descriptor).status, FinalGuestSurfaceStatus::Unsupported);
    descriptor = RouteDescriptor();
    descriptor.source_format = FinalGuestSurfaceFormat::Rgba16Float;
    EXPECT_EQ(PlanPpSourceReconstruction(descriptor).status, FinalGuestSurfaceStatus::Unsupported);
    descriptor = RouteDescriptor();
    descriptor.output_format = FinalGuestSurfaceFormat::A2R10G10B10;
    EXPECT_EQ(PlanPpSourceReconstruction(descriptor).status, FinalGuestSurfaceStatus::Unsupported);
}

TEST(PpSourceReconstruction, MissingResourcesAndOverflowFailClosedTransactionally) {
    auto descriptor = RouteDescriptor();
    descriptor.source_snapshot_available = false;
    auto plan = PlanPpSourceReconstruction(descriptor);
    EXPECT_EQ(plan.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(plan.source_copy_count, 0u);
    EXPECT_EQ(plan.reconstruction_draw_count, 0u);

    descriptor = RouteDescriptor();
    descriptor.reconstruction_output_available = false;
    EXPECT_EQ(PlanPpSourceReconstruction(descriptor).status,
              FinalGuestSurfaceStatus::InvalidationLoss);

    descriptor = RouteDescriptor();
    descriptor.slot_bytes = 15'794'175;
    plan = PlanPpSourceReconstruction(descriptor);
    EXPECT_EQ(plan.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_EQ(plan.present_copy_count, 0u);
    EXPECT_EQ(plan.reconstruction_draw_count, 0u);
}

TEST(PpSourceReconstruction, CaptureOrderHasExactBarriersAndNoPreSampleRepair) {
    const auto plan = PlanPpSourceReconstruction(RouteDescriptor());
    EXPECT_EQ(plan.source_image_barrier_count, 2u);
    EXPECT_EQ(plan.snapshot_image_barrier_count, 2u);
    EXPECT_EQ(plan.reconstruction_output_barrier_count, 2u);
    EXPECT_TRUE(plan.source_transition_occurs_after_visible_sample);
    EXPECT_TRUE(plan.reconstruction_restored_for_present_transfer);
    EXPECT_FALSE(plan.inserts_barrier_before_visible_sample);
}

TEST(PpSourceReconstruction, ClassificationSeparatesReproducedCleanAndAmbiguous) {
    EXPECT_EQ(ClassifyPpSourceReconstruction(true, true),
              PpSourceReconstructionClass::ReproducedFromSnapshot);
    EXPECT_EQ(ClassifyPpSourceReconstruction(true, false),
              PpSourceReconstructionClass::NotReproducedFromSnapshot);
    EXPECT_EQ(ClassifyPpSourceReconstruction(false, true), PpSourceReconstructionClass::Unassessed);
    EXPECT_EQ(ClassifyPpSourceReconstruction(false, false),
              PpSourceReconstructionClass::Unassessed);
}

TEST(PpSourceReconstruction, PrivacyFormatterContainsOnlyCalibratedOrdinals) {
    const PpSourceReconstructionReport report{
        .request_ordinal = 193,
        .a_sequence = 4262,
        .b_sequence = 4268,
        .c_sequence = 4273,
        .reproduced_ordinals = {1024},
        .not_reproduced_ordinals = {1299},
        .reproduced_count = 1,
        .not_reproduced_count = 1,
    };
    const auto line = FormatPpSourceReconstructionReport(report);
    EXPECT_EQ(line, "PPSR q=193 abc=4262/4268/4273 yes=1024 no=1299 amb= st=0");
    EXPECT_EQ(line.find("pixel"), std::string::npos);
    EXPECT_EQ(line.find("address"), std::string::npos);
    EXPECT_EQ(line.find("image"), std::string::npos);
    EXPECT_EQ(line.find("hash"), std::string::npos);
}

} // namespace
} // namespace Vulkan::HostPasses
