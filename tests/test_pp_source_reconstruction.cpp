// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/host_passes/pp_sampled_input.h"
#include "video_core/renderer_vulkan/host_passes/pp_source_reconstruction.h"
#include "video_core/renderer_vulkan/present_frame_transition.h"

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
        .bound_base_mip = 0,
        .bound_mip_count = 1,
        .bound_base_layer = 0,
        .bound_layer_count = 1,
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
    descriptor = RouteDescriptor();
    descriptor.bound_base_mip = 1;
    EXPECT_EQ(PlanPpSourceReconstruction(descriptor).status, FinalGuestSurfaceStatus::Unsupported);
    descriptor = RouteDescriptor();
    descriptor.bound_mip_count = 2;
    EXPECT_EQ(PlanPpSourceReconstruction(descriptor).status, FinalGuestSurfaceStatus::Unsupported);
    descriptor = RouteDescriptor();
    descriptor.bound_base_layer = 1;
    EXPECT_EQ(PlanPpSourceReconstruction(descriptor).status, FinalGuestSurfaceStatus::Unsupported);
    descriptor = RouteDescriptor();
    descriptor.bound_layer_count = 2;
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

TEST(PpSourceReconstruction, ExplicitStageIsCalibratedAndUsesSampledInputPipeline) {
    const auto config = ResolveFinalGuestSurfaceContentConfig([](std::string_view key) {
        if (key == "SHADPS4_FINAL_GUEST_SURFACE_CONTENT") {
            return std::optional<std::string_view>{"1"};
        }
        if (key == "SHADPS4_FINAL_GUEST_SURFACE_STAGE") {
            return std::optional<std::string_view>{"pp_source_reconstruction"};
        }
        if (key == "SHADPS4_FINAL_GUEST_SURFACE_CALIBRATED_TRIPLETS") {
            return std::optional<std::string_view>{"1"};
        }
        if (key == "SHADPS4_FINAL_GUEST_SURFACE_EXPECTED_CALIBRATIONS") {
            return std::optional<std::string_view>{"300"};
        }
        return std::optional<std::string_view>{};
    });
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->stage, FinalGuestSurfaceStage::PpSourceReconstruction);
    EXPECT_TRUE(IsPresentFinalGuestSurfaceStage(config->stage));
    EXPECT_FALSE(FinalGuestSurfaceLogPolicy(config->stage).verbose_frame_reports);
    EXPECT_EQ(PpDiagnosticModeForStage(config->stage), PpDiagnosticMode::SampledInput);
}

TEST(PpSourceReconstruction, AttachesFourthFullOutputPlaneWithinOneSlot) {
    auto output = PlanFinalGuestSurfaceTiles({
        .width = 1280,
        .height = 720,
        .depth = 1,
        .mip_levels = 1,
        .array_layers = 1,
        .samples = 1,
        .type = FinalGuestSurfaceImageType::Color2D,
        .aspect = FinalGuestSurfaceAspect::Color,
        .format = FinalGuestSurfaceFormat::Bgra8,
        .comparison = FinalGuestSurfaceComparison::LocalizedVisualReturn,
        .stage = FinalGuestSurfaceStage::PpSourceReconstruction,
        .logical_width = 1280,
        .logical_height = 720,
        .logical_full_fit = true,
        .logical_top_left = true,
        .logical_no_y_flip = true,
    });
    output.sample_bytes = ExistingPairedBytes;
    output.copy_region_count = 3;
    const auto reconstruction = PlanPpSourceReconstruction(RouteDescriptor());
    const auto plan = AttachPpSourceReconstructionPlane(output, reconstruction);
    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(plan.paired_reconstruction_offset, ExistingPairedBytes);
    EXPECT_EQ(plan.paired_reconstruction_bytes, 1280u * 720u * 4u);
    EXPECT_EQ(plan.paired_reconstruction_row_bytes, 1280u * 4u);
    EXPECT_EQ(plan.paired_reconstruction_format, FinalGuestSurfaceFormat::Bgra8);
    EXPECT_EQ(plan.sample_bytes, 15'794'176u);
    EXPECT_EQ(plan.copy_region_count, 4u);
}

TEST(PpSourceReconstruction, CalibratedReducerUsesExactPredicateOnReconstructionPlane) {
    constexpr u32 Width = 32;
    constexpr u32 Height = 32;
    auto plan = PlanFinalGuestSurfaceTiles({
        .width = Width,
        .height = Height,
        .depth = 1,
        .mip_levels = 1,
        .array_layers = 1,
        .samples = 1,
        .type = FinalGuestSurfaceImageType::Color2D,
        .aspect = FinalGuestSurfaceAspect::Color,
        .format = FinalGuestSurfaceFormat::Bgra8,
        .comparison = FinalGuestSurfaceComparison::LocalizedVisualReturn,
        .stage = FinalGuestSurfaceStage::PpSourceReconstruction,
        .logical_width = Width,
        .logical_height = Height,
        .logical_full_fit = true,
        .logical_top_left = true,
        .logical_no_y_flip = true,
    });
    const u32 output_bytes = Width * Height * 4;
    const auto reconstruction = PlanPpSourceReconstruction({
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
        .source_width = Width,
        .source_height = Height,
        .output_width = Width,
        .output_height = Height,
        .bound_mip_count = 1,
        .bound_layer_count = 1,
        .source_format = FinalGuestSurfaceFormat::Bgra8,
        .output_format = FinalGuestSurfaceFormat::Bgra8,
        .existing_readback_bytes = output_bytes,
        .slot_bytes = 16u << 20,
        .alignment = 16,
    });
    plan = AttachPpSourceReconstructionPlane(plan, reconstruction);
    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);

    const auto fill_plane = [&](std::vector<std::byte>& bytes, u32 offset, u8 value,
                                u32 pixels = Width * Height) {
        for (u32 pixel = 0; pixel < pixels; ++pixel) {
            const size_t at = offset + static_cast<size_t>(pixel) * 4;
            bytes[at] = bytes[at + 1] = bytes[at + 2] = std::byte{value};
        }
    };
    std::vector<std::byte> a(plan.sample_bytes);
    auto b = a;
    auto c = a;
    fill_plane(b, 0, 48, Width * Height / 4);
    fill_plane(b, plan.paired_reconstruction_offset, 48, Width * Height / 4);

    const auto selector = ParseFinalGuestSurfaceWatchOrdinals("1");
    const auto transport = FinalGuestSurfaceTransport{
        .surface_identity = 1,
        .backing_generation = 1,
        .format = FinalGuestSurfaceFormat::Bgra8,
        .width = Width,
        .height = Height,
    };
    const auto evaluate = [&](const std::vector<std::byte>& middle,
                              const std::vector<std::byte>& current) {
        FinalGuestSurfaceReducer reducer{FinalGuestSurfaceLagConfig::Defaults(), selector};
        (void)reducer.Observe(10, 1'000'000, transport, plan, a);
        (void)reducer.Observe(11, 1'100'000, transport, plan, middle);
        (void)reducer.Observe(12, 1'200'000, transport, plan, current);
        return reducer.EvaluateCalibratedTriplet({1, 10, 1'000'000, true}, {2, 11, 1'100'000, true},
                                                 {3, 12, 1'200'000, true}, true);
    };

    const auto reproduced = evaluate(b, c);
    ASSERT_TRUE(reproduced.has_value());
    ASSERT_EQ(reproduced->matched_ordinal_count, 1u);
    EXPECT_EQ(reproduced->reconstruction_reproduced_ordinal_count, 1u);
    EXPECT_EQ(reproduced->reconstruction_reproduced_ordinals[0], 1u);
    EXPECT_EQ(reproduced->reconstruction_not_reproduced_ordinal_count, 0u);

    auto stable_reconstruction_middle = b;
    fill_plane(stable_reconstruction_middle, plan.paired_reconstruction_offset, 0);
    const auto not_reproduced = evaluate(stable_reconstruction_middle, c);
    ASSERT_TRUE(not_reproduced.has_value());
    EXPECT_EQ(not_reproduced->matched_ordinal_count, 1u);
    EXPECT_EQ(not_reproduced->reconstruction_reproduced_ordinal_count, 0u);
    EXPECT_EQ(not_reproduced->reconstruction_not_reproduced_ordinal_count, 1u);
    EXPECT_EQ(not_reproduced->reconstruction_not_reproduced_ordinals[0], 1u);

    const auto text = FormatFinalGuestSurfaceCalibratedReport(*reproduced);
    EXPECT_NE(text.find(" ry=1"), std::string::npos);
    EXPECT_NE(text.find(" rn="), std::string::npos);
    EXPECT_EQ(text.find("pixel"), std::string::npos);
}

TEST(PpSourceReconstruction, PresentTransferAddsExactlyOneFourthPlane) {
    const auto transfer = PlanPpSampledInputTransfer(true, false, true, true, true);
    EXPECT_TRUE(transfer.copy);
    EXPECT_EQ(transfer.color_write_to_transfer_barriers, 3u);
    EXPECT_EQ(transfer.copy_regions, 4u);
    EXPECT_TRUE(transfer.paired_output_and_raw);
    EXPECT_TRUE(transfer.paired_source_backing_snapshot);
    EXPECT_TRUE(transfer.paired_source_reconstruction);
    EXPECT_TRUE(IsPpSampledInputTransferContractValid(transfer));

    const auto normal = PlanPpSampledInputTransfer(true, false, true, true, false);
    EXPECT_EQ(normal.copy_regions, 3u);
    EXPECT_FALSE(normal.paired_source_reconstruction);
    EXPECT_TRUE(IsPpSampledInputTransferContractValid(normal));
}

TEST(PpSourceReconstruction, PresentTransitionsReconstructionFromGeneralAfterColorWrite) {
    const auto transition = GetPpSourceReconstructionCaptureTransition(true);
    EXPECT_TRUE(transition.required);
    EXPECT_EQ(transition.src_stage, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    EXPECT_EQ(transition.src_access, vk::AccessFlagBits2::eColorAttachmentWrite);
    EXPECT_EQ(transition.dst_stage, vk::PipelineStageFlagBits2::eTransfer);
    EXPECT_EQ(transition.dst_access, vk::AccessFlagBits2::eTransferRead);
    EXPECT_EQ(transition.old_layout, vk::ImageLayout::eGeneral);
    EXPECT_EQ(transition.new_layout, vk::ImageLayout::eTransferSrcOptimal);
    EXPECT_FALSE(GetPpSourceReconstructionCaptureTransition(false).required);
}

TEST(PpSourceReconstruction, FlipPublicationSnapshotPrecedesVisiblePpAfterGuestRenderingEnds) {
    auto descriptor = RouteDescriptor();
    descriptor.snapshot_point = PpSourceSnapshotPoint::FlipPublication;
    const auto plan = PlanPpSourceReconstruction(descriptor);

    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_TRUE(plan.guest_rendering_ended_before_snapshot);
    EXPECT_TRUE(plan.source_snapshot_precedes_visible_draw);
    EXPECT_FALSE(plan.visible_draw_precedes_source_snapshot);
    EXPECT_TRUE(plan.source_snapshot_precedes_reconstruction_draw);
    EXPECT_TRUE(plan.snapshot_visibility_transition_can_perturb_visible_pp);
    EXPECT_EQ(plan.source_copy_count, 1u);
    EXPECT_EQ(plan.total_pp_draw_count, 2u);
    EXPECT_FALSE(plan.cpu_wait);
    EXPECT_FALSE(plan.finish);
}

TEST(PpSourceReconstruction, FlipPublicationStageIsExclusiveCalibratedAndPresentOwned) {
    const auto read = [](const char* name) -> std::optional<std::string_view> {
        const std::string_view key{name};
        if (key == "SHADPS4_FINAL_GUEST_SURFACE_CONTENT" ||
            key == "SHADPS4_FINAL_GUEST_SURFACE_CALIBRATED_TRIPLETS") {
            return "1";
        }
        if (key == "SHADPS4_FINAL_GUEST_SURFACE_STAGE") {
            return "pp_source_publication_reconstruction";
        }
        if (key == "SHADPS4_FINAL_GUEST_SURFACE_EXPECTED_CALIBRATIONS") {
            return "300";
        }
        return std::nullopt;
    };
    const auto config = ResolveFinalGuestSurfaceContentConfig(read);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->stage, FinalGuestSurfaceStage::PpSourcePublicationReconstruction);
    EXPECT_TRUE(config->calibrated_triplets);
    EXPECT_TRUE(IsPresentFinalGuestSurfaceStage(config->stage));
    EXPECT_FALSE(FinalGuestSurfaceLogPolicy(config->stage).verbose_frame_reports);
}

} // namespace
} // namespace Vulkan::HostPasses
