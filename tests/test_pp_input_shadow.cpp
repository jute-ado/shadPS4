// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include <string_view>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/final_guest_surface_content.h"
#include "video_core/renderer_vulkan/host_passes/pp_input_shadow.h"

namespace Vulkan {
namespace {

using HostPasses::PlanPpInputShadow;

TEST(PpInputShadow, DisabledPathCreatesNoDiagnosticResourcesPipelineOrCopies) {
    const auto plan = PlanPpInputShadow(false, 3);
    EXPECT_EQ(plan.shadow_pipeline_count, 0u);
    EXPECT_EQ(plan.shadow_shader_count, 0u);
    EXPECT_EQ(plan.shadow_image_count, 0u);
    EXPECT_EQ(plan.shadow_copy_count, 0u);
    EXPECT_EQ(plan.shadow_fragment_output_count, 0u);
    EXPECT_EQ(plan.shadow_color_attachment_count, 0u);
    EXPECT_FALSE(plan.requires_cpu_wait);
}

TEST(PpInputShadow, EnabledPathMirrorsOneInvocationIntoSameFormatAttachments) {
    const auto plan = PlanPpInputShadow(true, 3);
    EXPECT_EQ(plan.shadow_pipeline_count, 1u);
    EXPECT_EQ(plan.shadow_shader_count, 1u);
    EXPECT_EQ(plan.shadow_image_count, 3u);
    EXPECT_EQ(plan.shadow_copy_count, 1u);
    EXPECT_EQ(plan.shadow_fragment_output_count, 2u);
    EXPECT_EQ(plan.shadow_color_attachment_count, 2u);
    EXPECT_EQ(plan.draw_count, 1u);
    EXPECT_TRUE(plan.identical_computed_color);
    EXPECT_TRUE(plan.same_format_required);
    EXPECT_FALSE(plan.requires_cpu_wait);
}

TEST(PpInputShadow, StageIsExclusiveAndRequiresCalibratedPostProcessConfig) {
    const auto valid = ResolveFinalGuestSurfaceContentConfig(
        [](std::string_view name) -> std::optional<std::string_view> {
            if (name == "SHADPS4_FINAL_GUEST_SURFACE_CONTENT" ||
                name == "SHADPS4_FINAL_GUEST_SURFACE_CALIBRATED_TRIPLETS") {
                return "1";
            }
            if (name == "SHADPS4_FINAL_GUEST_SURFACE_STAGE") {
                return "pp_input_shadow";
            }
            if (name == "SHADPS4_FINAL_GUEST_SURFACE_EXPECTED_CALIBRATIONS") {
                return "300";
            }
            return std::nullopt;
        });
    ASSERT_TRUE(valid.has_value());
    EXPECT_EQ(valid->stage, FinalGuestSurfaceStage::PpInputShadow);
    EXPECT_TRUE(valid->calibrated_triplets);
    EXPECT_TRUE(IsPresentFinalGuestSurfaceStage(valid->stage));

    const auto missing_calibration = ResolveFinalGuestSurfaceContentConfig(
        [](std::string_view name) -> std::optional<std::string_view> {
            if (name == "SHADPS4_FINAL_GUEST_SURFACE_CONTENT") {
                return "1";
            }
            if (name == "SHADPS4_FINAL_GUEST_SURFACE_STAGE") {
                return "pp_input_shadow";
            }
            return std::nullopt;
        });
    EXPECT_FALSE(missing_calibration.has_value());
}

TEST(PpInputShadow, MetadataIsExactStableAndGenerationTracked) {
    FinalGuestSurfacePpInputConfigTracker tracker;
    FinalGuestSurfacePpInputDescriptor descriptor{
        .fsr_enabled = true,
        .input_width = 1920,
        .input_height = 1080,
        .output_width = 1280,
        .output_height = 720,
        .source_format = FinalGuestSurfaceFormat::Bgra8,
        .output_format = FinalGuestSurfaceFormat::Bgra8,
        .gamma_bits = std::bit_cast<u32>(1.0f),
        .pp_hdr = false,
        .frame_hdr = false,
    };
    const auto first = tracker.Observe(descriptor);
    EXPECT_TRUE(first.valid);
    EXPECT_TRUE(first.fsr_bypassed);
    EXPECT_EQ(first.source_width, 1920u);
    EXPECT_EQ(first.source_height, 1080u);
    EXPECT_EQ(first.output_width, 1280u);
    EXPECT_EQ(first.output_height, 720u);
    EXPECT_EQ(first.source_format, FinalGuestSurfaceFormat::Bgra8);
    EXPECT_EQ(first.output_format, FinalGuestSurfaceFormat::Bgra8);
    EXPECT_EQ(first.gamma_bits, std::bit_cast<u32>(1.0f));
    EXPECT_FALSE(first.hdr);
    EXPECT_EQ(first.config_generation, 1u);
    EXPECT_EQ(tracker.Observe(descriptor), first);

    descriptor.gamma_bits = std::bit_cast<u32>(1.1f);
    const auto changed = tracker.Observe(descriptor);
    EXPECT_TRUE(changed.valid);
    EXPECT_EQ(changed.config_generation, 2u);
    EXPECT_NE(changed, first);
}

TEST(PpInputShadow, ActiveFsrInvalidFormatsExtentAndHdrMismatchFailClosed) {
    FinalGuestSurfacePpInputConfigTracker tracker;
    FinalGuestSurfacePpInputDescriptor descriptor{
        .fsr_enabled = true,
        .input_width = 640,
        .input_height = 360,
        .output_width = 1280,
        .output_height = 720,
        .source_format = FinalGuestSurfaceFormat::Bgra8,
        .output_format = FinalGuestSurfaceFormat::Bgra8,
        .gamma_bits = std::bit_cast<u32>(1.0f),
    };
    EXPECT_FALSE(tracker.Observe(descriptor).valid);

    descriptor.fsr_enabled = false;
    descriptor.source_format = FinalGuestSurfaceFormat::Unsupported;
    EXPECT_FALSE(tracker.Observe(descriptor).valid);
    descriptor.source_format = FinalGuestSurfaceFormat::Bgra8;
    descriptor.input_width = 0;
    EXPECT_FALSE(tracker.Observe(descriptor).valid);
    descriptor.input_width = 1920;
    descriptor.pp_hdr = true;
    descriptor.frame_hdr = false;
    EXPECT_FALSE(tracker.Observe(descriptor).valid);
    descriptor.frame_hdr = true;
    descriptor.output_format = FinalGuestSurfaceFormat::A2R10G10B10;
    EXPECT_FALSE(tracker.Observe(descriptor).valid);
}

TEST(PpInputShadow, FrameTokenTransfersOnceAndReuseOrStaleStateFailsClosed) {
    FinalGuestSurfacePpInputFrameState state;
    FinalGuestSurfacePpInputMetadata metadata{.config_generation = 7, .valid = true};
    EXPECT_EQ(state.Assign({100, 2'000'000, 11, metadata}), FinalGuestSurfaceStatus::Complete);
    const auto first = state.TakeForPresent(false);
    ASSERT_TRUE(first.emit);
    EXPECT_EQ(first.payload.sequence, 100u);
    EXPECT_EQ(first.payload.process_time_us, 2'000'000u);
    EXPECT_EQ(first.payload.token, 11u);
    EXPECT_EQ(first.payload.metadata, metadata);
    EXPECT_EQ(first.status, FinalGuestSurfaceStatus::Complete);

    const auto stale = state.TakeForPresent(false);
    EXPECT_FALSE(stale.emit);
    EXPECT_EQ(stale.status, FinalGuestSurfaceStatus::AlreadyConsumed);

    EXPECT_EQ(state.Assign({101, 2'025'000, 12, metadata}), FinalGuestSurfaceStatus::Complete);
    const auto reused = state.TakeForPresent(true);
    EXPECT_FALSE(reused.emit);
    EXPECT_EQ(reused.status, FinalGuestSurfaceStatus::GapLoss);

    EXPECT_EQ(state.Assign({102, 2'050'000, 13, metadata}), FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(state.Assign({103, 2'075'000, 14, metadata}), FinalGuestSurfaceStatus::GapLoss);
    EXPECT_FALSE(state.TakeForPresent(false).emit);
}

TEST(PpInputShadow, TransferContractIsOneDeferredCopyWithoutWaitOrPointerLifetime) {
    const auto enabled = PlanPpInputShadowTransfer(true, false, true, true);
    EXPECT_TRUE(enabled.copy);
    EXPECT_EQ(enabled.color_write_to_transfer_barriers, 1u);
    EXPECT_EQ(enabled.copy_regions, 1u);
    EXPECT_FALSE(enabled.cpu_wait);
    EXPECT_FALSE(enabled.finish);
    EXPECT_FALSE(enabled.callback_retains_frame);
    EXPECT_FALSE(enabled.callback_retains_image);
    EXPECT_FALSE(enabled.callback_retains_vk_image);

    EXPECT_FALSE(PlanPpInputShadowTransfer(false, false, true, true).copy);
    EXPECT_FALSE(PlanPpInputShadowTransfer(true, true, true, true).copy);
    EXPECT_FALSE(PlanPpInputShadowTransfer(true, false, false, true).copy);
    EXPECT_FALSE(PlanPpInputShadowTransfer(true, false, true, false).copy);
}

TEST(PpInputShadow, TilePlanUsesCalibratedLocalizedOutputLattice) {
    const auto plan = PlanFinalGuestSurfaceTiles({
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
        .stage = FinalGuestSurfaceStage::PpInputShadow,
        .logical_width = 1280,
        .logical_height = 720,
        .logical_full_fit = true,
        .logical_top_left = true,
        .logical_no_y_flip = true,
    });
    EXPECT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(plan.stage, FinalGuestSurfaceStage::PpInputShadow);
    EXPECT_EQ(plan.tile_count, 3476u);
    EXPECT_EQ(plan.copy_region_count, 1u);
}

TEST(PpInputShadow, NormalPostProcessInvocationRemainsSingleOutputAndAttachment) {
    const auto invocation = HostPasses::PlanPostProcessingInvocation(
        false, false, FinalGuestSurfaceFormat::Bgra8, FinalGuestSurfaceFormat::Unsupported);
    EXPECT_TRUE(invocation.draw_normal_output);
    EXPECT_FALSE(invocation.draw_shadow_output);
    EXPECT_EQ(invocation.pipeline, HostPasses::PpPipelineSelection::Normal);
    EXPECT_EQ(invocation.fragment_output_count, 1u);
    EXPECT_EQ(invocation.color_attachment_count, 1u);
    EXPECT_EQ(invocation.attachment_formats[0], FinalGuestSurfaceFormat::Bgra8);
    EXPECT_EQ(invocation.draw_count, 1u);
    EXPECT_EQ(invocation.status, FinalGuestSurfaceStatus::Complete);
}

TEST(PpInputShadow, SelectedShadowInvocationUsesDualPipelineAndSameFormatTwice) {
    const auto invocation = HostPasses::PlanPostProcessingInvocation(
        true, true, FinalGuestSurfaceFormat::Bgra8, FinalGuestSurfaceFormat::Bgra8);
    EXPECT_TRUE(invocation.draw_normal_output);
    EXPECT_TRUE(invocation.draw_shadow_output);
    EXPECT_EQ(invocation.pipeline, HostPasses::PpPipelineSelection::ShadowDualOutput);
    EXPECT_EQ(invocation.fragment_output_count, 2u);
    EXPECT_EQ(invocation.color_attachment_count, 2u);
    EXPECT_EQ(invocation.attachment_formats[0], FinalGuestSurfaceFormat::Bgra8);
    EXPECT_EQ(invocation.attachment_formats[1], FinalGuestSurfaceFormat::Bgra8);
    EXPECT_EQ(invocation.draw_count, 1u);
    EXPECT_TRUE(invocation.identical_computed_color);
    EXPECT_EQ(invocation.status, FinalGuestSurfaceStatus::Complete);

    const auto mismatch = HostPasses::PlanPostProcessingInvocation(
        true, true, FinalGuestSurfaceFormat::Bgra8, FinalGuestSurfaceFormat::Rgba8);
    EXPECT_TRUE(mismatch.draw_normal_output);
    EXPECT_FALSE(mismatch.draw_shadow_output);
    EXPECT_EQ(mismatch.pipeline, HostPasses::PpPipelineSelection::Normal);
    EXPECT_EQ(mismatch.status, FinalGuestSurfaceStatus::Unsupported);

    const auto absent = HostPasses::PlanPostProcessingInvocation(
        true, false, FinalGuestSurfaceFormat::Bgra8, FinalGuestSurfaceFormat::Bgra8);
    EXPECT_TRUE(absent.draw_normal_output);
    EXPECT_FALSE(absent.draw_shadow_output);
    EXPECT_EQ(absent.pipeline, HostPasses::PpPipelineSelection::Normal);
    EXPECT_EQ(absent.status, FinalGuestSurfaceStatus::InvalidationLoss);
}

TEST(PpInputShadow, PresentHandoffQueuesContentBeforeCalibrationWithoutDrawSchedulerCallback) {
    const auto plan = PlanPpInputShadowPresentHandoff(true, false, true, true);
    EXPECT_TRUE(plan.copy);
    EXPECT_EQ(plan.scheduler, FinalGuestSurfaceDeferredScheduler::Present);
    EXPECT_EQ(plan.content_callback_order, 1u);
    EXPECT_EQ(plan.calibration_callback_order, 2u);
    EXPECT_LT(plan.content_callback_order, plan.calibration_callback_order);
    EXPECT_FALSE(plan.defer_on_draw_scheduler);
    EXPECT_TRUE(plan.callback_payload_is_scalar_only);
    EXPECT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);

    const auto busy = PlanPpInputShadowPresentHandoff(true, false, true, false);
    EXPECT_FALSE(busy.copy);
    EXPECT_EQ(busy.scheduler, FinalGuestSurfaceDeferredScheduler::Present);
    EXPECT_EQ(busy.content_callback_order, 1u);
    EXPECT_EQ(busy.calibration_callback_order, 2u);
    EXPECT_EQ(busy.status, FinalGuestSurfaceStatus::BusyLoss);

    EXPECT_FALSE(PlanPpInputShadowPresentHandoff(true, true, true, true).copy);
    EXPECT_FALSE(PlanPpInputShadowPresentHandoff(true, false, false, true).copy);
}

TEST(PpInputShadow, StageUsesCalibratedOnlyCompactOutputWithExplicitCoverage) {
    const auto shadow = FinalGuestSurfaceLogPolicy(FinalGuestSurfaceStage::PpInputShadow);
    EXPECT_FALSE(shadow.verbose_frame_reports);
    EXPECT_TRUE(shadow.calibrated_triplet_reports);
    EXPECT_TRUE(shadow.stage_content_coverage);
    EXPECT_TRUE(shadow.calibrated_coverage);

    const auto post_pp = FinalGuestSurfaceLogPolicy(FinalGuestSurfaceStage::PostPp);
    EXPECT_TRUE(post_pp.verbose_frame_reports);
}

} // namespace
} // namespace Vulkan
