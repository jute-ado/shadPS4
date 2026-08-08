// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include <cstring>
#include <fstream>
#include <string_view>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/final_guest_surface_content.h"
#include "video_core/renderer_vulkan/host_passes/pp_sampled_input.h"

namespace Vulkan {
namespace {

using HostPasses::PlanPpSampledInput;
using HostPasses::PlanPpSampledInputInvocation;

void WriteHalfRgba(std::vector<std::byte>& bytes, u32 pixel, u16 red, u16 green, u16 blue,
                   u16 alpha = 0x3c00) {
    const std::array values{red, green, blue, alpha};
    std::memcpy(bytes.data() + static_cast<size_t>(pixel) * sizeof(values), values.data(),
                sizeof(values));
}

TEST(PpSampledInput, DisabledPathCreatesNoResourcesPipelineCopiesOrWaits) {
    const auto plan = PlanPpSampledInput(false, 3);
    EXPECT_EQ(plan.pipeline_count, 0u);
    EXPECT_EQ(plan.shader_count, 0u);
    EXPECT_EQ(plan.float_image_count, 0u);
    EXPECT_EQ(plan.copy_count, 0u);
    EXPECT_EQ(plan.fragment_output_count, 0u);
    EXPECT_EQ(plan.color_attachment_count, 0u);
    EXPECT_FALSE(plan.cpu_wait);
    EXPECT_FALSE(plan.finish);
}

TEST(PpSampledInput, OneInvocationWritesNormalComputedAndRawFloatSample) {
    const auto plan = PlanPpSampledInput(true, 3);
    EXPECT_EQ(plan.pipeline_count, 1u);
    EXPECT_EQ(plan.shader_count, 1u);
    EXPECT_EQ(plan.float_image_count, 3u);
    EXPECT_EQ(plan.copy_count, 1u);
    EXPECT_EQ(plan.fragment_output_count, 2u);
    EXPECT_EQ(plan.color_attachment_count, 2u);
    EXPECT_EQ(plan.draw_count, 1u);
    EXPECT_EQ(plan.texture_sample_count, 1u);
    EXPECT_TRUE(plan.normal_output_is_computed_color);
    EXPECT_TRUE(plan.diagnostic_output_is_raw_linear_sample);
    EXPECT_EQ(plan.diagnostic_format, FinalGuestSurfaceFormat::Rgba16Float);

    const auto invocation = PlanPpSampledInputInvocation(true, true, FinalGuestSurfaceFormat::Bgra8,
                                                         FinalGuestSurfaceFormat::Rgba16Float);
    EXPECT_EQ(invocation.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(invocation.attachment_formats[0], FinalGuestSurfaceFormat::Bgra8);
    EXPECT_EQ(invocation.attachment_formats[1], FinalGuestSurfaceFormat::Rgba16Float);
    EXPECT_EQ(invocation.color_attachment_count, 2u);
    EXPECT_EQ(invocation.draw_count, 1u);
    EXPECT_TRUE(invocation.raw_sample_output);

    EXPECT_EQ(PlanPpSampledInputInvocation(true, true, FinalGuestSurfaceFormat::Bgra8,
                                           FinalGuestSurfaceFormat::Bgra8)
                  .status,
              FinalGuestSurfaceStatus::Unsupported);
}

TEST(PpSampledInput, StageIsExclusiveAndRequiresCalibratedTriplets) {
    const auto valid = ResolveFinalGuestSurfaceContentConfig(
        [](std::string_view name) -> std::optional<std::string_view> {
            if (name == "SHADPS4_FINAL_GUEST_SURFACE_CONTENT" ||
                name == "SHADPS4_FINAL_GUEST_SURFACE_CALIBRATED_TRIPLETS") {
                return "1";
            }
            if (name == "SHADPS4_FINAL_GUEST_SURFACE_STAGE") {
                return "pp_sampled_input";
            }
            if (name == "SHADPS4_FINAL_GUEST_SURFACE_EXPECTED_CALIBRATIONS") {
                return "300";
            }
            return std::nullopt;
        });
    ASSERT_TRUE(valid.has_value());
    EXPECT_EQ(valid->stage, FinalGuestSurfaceStage::PpSampledInput);
    EXPECT_TRUE(valid->calibrated_triplets);
    EXPECT_TRUE(IsPresentFinalGuestSurfaceStage(valid->stage));
    EXPECT_FALSE(FinalGuestSurfaceLogPolicy(valid->stage).verbose_frame_reports);
}

TEST(PpSampledInput, MetadataPreservesResolvedViewAndExactSettingsSnapshot) {
    FinalGuestSurfaceSampledInputConfigTracker tracker;
    FinalGuestSurfaceSampledInputDescriptor descriptor{
        .fsr_enabled = true,
        .input_width = 1920,
        .input_height = 1080,
        .output_width = 1280,
        .output_height = 720,
        .source_format = FinalGuestSurfaceFormat::Bgra8,
        .output_format = FinalGuestSurfaceFormat::Bgra8,
        .resolved_base_mip = 2,
        .resolved_mip_count = 1,
        .resolved_base_layer = 3,
        .resolved_layer_count = 1,
        .source_image_uid = 41,
        .source_backing_generation = 7,
        .gamma_bits = std::bit_cast<u32>(1.0f),
        .source_view_srgb = true,
        .settings_snapshot_matches_push = true,
    };
    const auto first = tracker.Observe(descriptor);
    ASSERT_TRUE(first.valid);
    EXPECT_TRUE(first.fsr_bypassed);
    EXPECT_EQ(first.resolved_base_mip, 2u);
    EXPECT_EQ(first.resolved_mip_count, 1u);
    EXPECT_EQ(first.resolved_base_layer, 3u);
    EXPECT_EQ(first.resolved_layer_count, 1u);
    EXPECT_EQ(first.source_image_uid, 41u);
    EXPECT_EQ(first.source_backing_generation, 7u);
    EXPECT_EQ(first.gamma_bits, std::bit_cast<u32>(1.0f));
    EXPECT_EQ(first.config_generation, 1u);

    descriptor.source_image_uid = 42;
    descriptor.source_backing_generation = 8;
    const auto rotated = tracker.Observe(descriptor);
    ASSERT_TRUE(rotated.valid);
    EXPECT_EQ(rotated.config_generation, 1u)
        << "physical source rotation must not break the logical sampled stream";
    EXPECT_EQ(rotated.source_image_uid, 42u);
    EXPECT_EQ(rotated.source_backing_generation, 8u);

    descriptor.resolved_base_mip = 4;
    EXPECT_EQ(tracker.Observe(descriptor).config_generation, 2u);
}

TEST(PpSampledInput, ActiveFsrHdrInvalidViewOrMismatchedSettingsFailClosed) {
    FinalGuestSurfaceSampledInputConfigTracker tracker;
    FinalGuestSurfaceSampledInputDescriptor descriptor{
        .fsr_enabled = true,
        .input_width = 640,
        .input_height = 360,
        .output_width = 1280,
        .output_height = 720,
        .source_format = FinalGuestSurfaceFormat::Bgra8,
        .output_format = FinalGuestSurfaceFormat::Bgra8,
        .resolved_mip_count = 1,
        .resolved_layer_count = 1,
        .source_image_uid = 1,
        .source_backing_generation = 1,
        .gamma_bits = std::bit_cast<u32>(1.0f),
        .source_view_srgb = true,
        .settings_snapshot_matches_push = true,
    };
    EXPECT_FALSE(tracker.Observe(descriptor).valid);
    descriptor.fsr_enabled = false;
    descriptor.pp_hdr = descriptor.frame_hdr = true;
    EXPECT_FALSE(tracker.Observe(descriptor).valid);
    descriptor.pp_hdr = descriptor.frame_hdr = false;
    descriptor.source_view_srgb = false;
    EXPECT_FALSE(tracker.Observe(descriptor).valid);
    descriptor.source_view_srgb = true;
    descriptor.settings_snapshot_matches_push = false;
    EXPECT_FALSE(tracker.Observe(descriptor).valid);
    descriptor.settings_snapshot_matches_push = true;
    descriptor.resolved_mip_count = 0;
    EXPECT_FALSE(tracker.Observe(descriptor).valid);
}

TEST(PpSampledInput, FloatTilePlanIsOneBoundedCopyOnLogicalOutputLattice) {
    const auto plan = PlanFinalGuestSurfaceTiles({
        .width = 1280,
        .height = 720,
        .depth = 1,
        .mip_levels = 1,
        .array_layers = 1,
        .samples = 1,
        .type = FinalGuestSurfaceImageType::Color2D,
        .aspect = FinalGuestSurfaceAspect::Color,
        .format = FinalGuestSurfaceFormat::Rgba16Float,
        .comparison = FinalGuestSurfaceComparison::SampledLinearVisualReturn,
        .stage = FinalGuestSurfaceStage::PpSampledInput,
        .logical_width = 1280,
        .logical_height = 720,
        .logical_full_fit = true,
        .logical_top_left = true,
        .logical_no_y_flip = true,
        .comparison_gamma_bits = std::bit_cast<u32>(1.0f),
    });
    EXPECT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(plan.tile_count, 3476u);
    EXPECT_EQ(plan.copy_region_count, 1u);
    EXPECT_EQ(plan.sample_bytes, 1280u * 720u * 8u);
    EXPECT_EQ(plan.comparison_gamma_bits, std::bit_cast<u32>(1.0f));
}

TEST(PpSampledInput, LinearHalfSamplesUseVisibleGammaPredicateAndIgnoreAlpha) {
    const auto selector = ParseFinalGuestSurfaceWatchOrdinals("1");
    const auto descriptor = FinalGuestSurfaceDescriptor{
        .width = 32,
        .height = 32,
        .depth = 1,
        .mip_levels = 1,
        .array_layers = 1,
        .samples = 1,
        .type = FinalGuestSurfaceImageType::Color2D,
        .aspect = FinalGuestSurfaceAspect::Color,
        .format = FinalGuestSurfaceFormat::Rgba16Float,
        .comparison = FinalGuestSurfaceComparison::SampledLinearVisualReturn,
        .stage = FinalGuestSurfaceStage::PpSampledInput,
        .logical_width = 32,
        .logical_height = 32,
        .logical_full_fit = true,
        .logical_top_left = true,
        .logical_no_y_flip = true,
        .comparison_gamma_bits = std::bit_cast<u32>(1.0f),
    };
    const auto plan = PlanFinalGuestSurfaceTiles(descriptor);
    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    std::vector<std::byte> a(plan.sample_bytes);
    for (u32 pixel = 0; pixel < 1024; ++pixel) {
        WriteHalfRgba(a, pixel, 0x3400, 0x3400, 0x3400);
    }
    auto b = a;
    auto c = a;
    for (u32 pixel = 0; pixel < 256; ++pixel) {
        WriteHalfRgba(b, pixel, 0x3c00, 0x3c00, 0x3c00);
    }
    FinalGuestSurfaceReducer reducer{FinalGuestSurfaceLagConfig::Defaults(), selector};
    const auto transport = FinalGuestSurfaceTransport{
        .surface_identity = 1,
        .backing_generation = 1,
        .format = FinalGuestSurfaceFormat::Rgba16Float,
        .width = 32,
        .height = 32,
    };
    (void)reducer.Observe(1, 1'000'000, transport, plan, a);
    (void)reducer.Observe(2, 1'100'000, transport, plan, b);
    const auto returned = reducer.Observe(3, 1'200'000, transport, plan, c);
    EXPECT_TRUE(returned.localized_aba);
    EXPECT_EQ(returned.aba_tile_ordinal_count, 1u);
    EXPECT_EQ(returned.aba_tile_ordinals[0], 1u);

    auto alpha_only = a;
    for (u32 pixel = 0; pixel < 1024; ++pixel) {
        WriteHalfRgba(alpha_only, pixel, 0x3400, 0x3400, 0x3400, 0x3c01);
    }
    FinalGuestSurfaceReducer alpha_reducer{FinalGuestSurfaceLagConfig::Defaults(), selector};
    (void)alpha_reducer.Observe(1, 1'000'000, transport, plan, a);
    (void)alpha_reducer.Observe(2, 1'100'000, transport, plan, alpha_only);
    EXPECT_FALSE(alpha_reducer.Observe(3, 1'200'000, transport, plan, a).localized_aba);
}

TEST(PpSampledInput, InvalidOrUnstampedFramesAreNeverAssigned) {
    EXPECT_FALSE(ShouldAssignPpSampledInputFrame(false, true, true));
    EXPECT_FALSE(ShouldAssignPpSampledInputFrame(true, false, true));
    EXPECT_FALSE(ShouldAssignPpSampledInputFrame(true, true, false));
    EXPECT_TRUE(ShouldAssignPpSampledInputFrame(true, true, true));
}

TEST(PpSampledInput, ProductionShaderWritesOneRawSampleBeforeGamma) {
    std::ifstream shader{PP_SAMPLED_INPUT_SHADER_PATH, std::ios::binary};
    ASSERT_TRUE(shader) << PP_SAMPLED_INPUT_SHADER_PATH;
    const std::string source{std::istreambuf_iterator<char>{shader},
                             std::istreambuf_iterator<char>{}};
    EXPECT_NE(source.find("layout (location = 0) out vec4 color;"), std::string::npos);
    EXPECT_NE(source.find("layout (location = 1) out vec4 sampled_linear;"), std::string::npos);
    EXPECT_NE(source.find("vec4 color_linear = texture(texSampler, uv);"), std::string::npos);
    EXPECT_NE(source.find("color = computed_color;"), std::string::npos);
    EXPECT_NE(source.find("sampled_linear = color_linear;"), std::string::npos);
    const auto sample = source.find("texture(texSampler, uv)");
    ASSERT_NE(sample, std::string::npos);
    EXPECT_EQ(source.find("texture(texSampler, uv)", sample + 1), std::string::npos);
}

TEST(PpSampledInput, ProductionStageSelectionIsExclusive) {
    EXPECT_EQ(PpDiagnosticModeForStage(FinalGuestSurfaceStage::GuestPreFsr),
              HostPasses::PpDiagnosticMode::None);
    EXPECT_EQ(PpDiagnosticModeForStage(FinalGuestSurfaceStage::PostPp),
              HostPasses::PpDiagnosticMode::None);
    EXPECT_EQ(PpDiagnosticModeForStage(FinalGuestSurfaceStage::PpInputShadow),
              HostPasses::PpDiagnosticMode::ComputedShadow);
    EXPECT_EQ(PpDiagnosticModeForStage(FinalGuestSurfaceStage::PpSampledInput),
              HostPasses::PpDiagnosticMode::SampledInput);
}

TEST(PpSampledInput, PresenterPreservesResolvedMipAndLayerWhileOverridingViewFormat) {
    const auto plan = PlanPpSampledInputSourceView({
        .resolved_base_mip = 2,
        .resolved_mip_count = 1,
        .resolved_base_layer = 3,
        .resolved_layer_count = 1,
        .requested_format = FinalGuestSurfaceFormat::Bgra8,
        .force_alpha_one = true,
    });
    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(plan.base_mip, 2u);
    EXPECT_EQ(plan.mip_count, 1u);
    EXPECT_EQ(plan.base_layer, 3u);
    EXPECT_EQ(plan.layer_count, 1u);
    EXPECT_EQ(plan.format, FinalGuestSurfaceFormat::Bgra8);
    EXPECT_TRUE(plan.force_alpha_one);

    EXPECT_EQ(PlanPpSampledInputSourceView({.resolved_mip_count = 0,
                                            .resolved_layer_count = 1,
                                            .requested_format = FinalGuestSurfaceFormat::Bgra8})
                  .status,
              FinalGuestSurfaceStatus::Unsupported);
}

TEST(PpSampledInput, ValidStampedFrameTransfersOnceWithoutStartupPoisoning) {
    FinalGuestSurfaceSampledInputFrameState state;
    FinalGuestSurfaceSampledInputMetadata metadata{
        .config_generation = 1,
        .valid = true,
    };
    EXPECT_EQ(state.AssignIfValid(false, {0, 0, 0, metadata}),
              FinalGuestSurfaceStatus::AlreadyConsumed);
    EXPECT_EQ(state.TakeForPresent(false).status, FinalGuestSurfaceStatus::AlreadyConsumed);

    EXPECT_EQ(state.AssignIfValid(true, {100, 2'000'000, 7, metadata}),
              FinalGuestSurfaceStatus::Complete);
    const auto first = state.TakeForPresent(false);
    ASSERT_TRUE(first.emit);
    EXPECT_EQ(first.payload.sequence, 100u);
    EXPECT_EQ(first.payload.token, 7u);
    EXPECT_EQ(first.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(state.TakeForPresent(false).status, FinalGuestSurfaceStatus::AlreadyConsumed);

    EXPECT_EQ(state.AssignIfValid(true, {101, 2'100'000, 8, metadata}),
              FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(state.TakeForPresent(true).status, FinalGuestSurfaceStatus::GapLoss);
}

TEST(PpSampledInput, PresentTransferIsOneFloatCopyWithScalarCallbackAndNoWait) {
    const auto plan = PlanPpSampledInputTransfer(true, false, true, true);
    EXPECT_TRUE(plan.copy);
    EXPECT_EQ(plan.color_write_to_transfer_barriers, 1u);
    EXPECT_EQ(plan.copy_regions, 1u);
    EXPECT_EQ(plan.format, FinalGuestSurfaceFormat::Rgba16Float);
    EXPECT_TRUE(plan.callback_payload_is_scalar_only);
    EXPECT_FALSE(plan.cpu_wait);
    EXPECT_FALSE(plan.finish);
    EXPECT_FALSE(plan.callback_retains_frame);
    EXPECT_FALSE(plan.callback_retains_image);
    EXPECT_FALSE(plan.callback_retains_vk_image);
    EXPECT_FALSE(PlanPpSampledInputTransfer(true, true, true, true).copy);
}

TEST(PpSampledInput, DiagnosticPreservesBaselineViewAndFailsClosedOnResolvedMismatch) {
    const auto baseline = AssessPpSampledInputSourceView({
        .resolved_base_mip = 0,
        .resolved_mip_count = 1,
        .resolved_base_layer = 0,
        .resolved_layer_count = 1,
        .bound_base_mip = 0,
        .bound_mip_count = 1,
        .bound_base_layer = 0,
        .bound_layer_count = 1,
    });
    EXPECT_EQ(baseline.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_TRUE(baseline.same_view_as_baseline);
    EXPECT_FALSE(baseline.diagnostic_changes_bound_view);
    EXPECT_FALSE(baseline.resolved_range_mismatch);

    const auto mismatch = AssessPpSampledInputSourceView({
        .resolved_base_mip = 2,
        .resolved_mip_count = 1,
        .resolved_base_layer = 3,
        .resolved_layer_count = 1,
        .bound_base_mip = 0,
        .bound_mip_count = 1,
        .bound_base_layer = 0,
        .bound_layer_count = 1,
    });
    EXPECT_EQ(mismatch.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_TRUE(mismatch.same_view_as_baseline);
    EXPECT_FALSE(mismatch.diagnostic_changes_bound_view);
    EXPECT_TRUE(mismatch.resolved_range_mismatch);
}

TEST(PpSampledInput, PairedCaptureUsesOneSlotForExactOutputAndRawSamplePlanes) {
    constexpr u64 SlotBytes = 16ull * 1024 * 1024;
    const auto plan = PlanPpSampledInputPairedCapture({
        .enabled = true,
        .width = 1280,
        .height = 720,
        .output_format = FinalGuestSurfaceFormat::Bgra8,
        .sampled_format = FinalGuestSurfaceFormat::Rgba16Float,
        .slot_bytes = SlotBytes,
        .alignment = 256,
    });
    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(plan.slot_count, 1u);
    EXPECT_EQ(plan.copy_region_count, 2u);
    EXPECT_EQ(plan.output_offset, 0u);
    EXPECT_EQ(plan.output_bytes, 1280u * 720u * 4u);
    EXPECT_EQ(plan.sampled_bytes, 1280u * 720u * 8u);
    EXPECT_GE(plan.sampled_offset, plan.output_bytes);
    EXPECT_LE(plan.total_bytes, SlotBytes);
    EXPECT_EQ(plan.output_comparison, FinalGuestSurfaceComparison::LocalizedVisualReturn);
    EXPECT_TRUE(plan.output_is_authoritative);
    EXPECT_FALSE(plan.cpu_gamma_reconstruction_is_authoritative);
    EXPECT_TRUE(plan.callback_payload_is_scalar_only);
    EXPECT_FALSE(plan.cpu_wait);
    EXPECT_FALSE(plan.finish);

    EXPECT_EQ(
        PlanPpSampledInputPairedCapture({
                                            .enabled = true,
                                            .width = 1920,
                                            .height = 1080,
                                            .output_format = FinalGuestSurfaceFormat::Bgra8,
                                            .sampled_format = FinalGuestSurfaceFormat::Rgba16Float,
                                            .slot_bytes = SlotBytes,
                                            .alignment = 256,
                                        })
            .status,
        FinalGuestSurfaceStatus::CapacityLoss);
}

TEST(PpSampledInput, PairClassificationNeverSubstitutesCpuGammaForExactOutput) {
    EXPECT_EQ(ClassifyPpSampledInputPair({.output_visual_return = false,
                                          .raw_sample_return = true,
                                          .raw_sample_stable = false,
                                          .complete = true}),
              PpSampledInputBoundary::OutputClean);
    EXPECT_EQ(ClassifyPpSampledInputPair({.output_visual_return = true,
                                          .raw_sample_return = true,
                                          .raw_sample_stable = false,
                                          .complete = true}),
              PpSampledInputBoundary::AtOrBeforeSample);
    EXPECT_EQ(ClassifyPpSampledInputPair({.output_visual_return = true,
                                          .raw_sample_return = false,
                                          .raw_sample_stable = true,
                                          .complete = true}),
              PpSampledInputBoundary::AfterSample);
    EXPECT_EQ(ClassifyPpSampledInputPair({.output_visual_return = true,
                                          .raw_sample_return = false,
                                          .raw_sample_stable = false,
                                          .complete = false}),
              PpSampledInputBoundary::Ambiguous);
}

TEST(PpSampledInput, SelectedInvalidFramesEmitExplicitLossWhileStartupIsSilent) {
    const auto startup = PlanPpSampledInputObservation(
        {.in_window = false, .stamp_valid = false, .metadata_valid = false});
    EXPECT_FALSE(startup.emit);
    EXPECT_EQ(startup.status, FinalGuestSurfaceStatus::AlreadyConsumed);

    const auto missing_stamp = PlanPpSampledInputObservation(
        {.in_window = true, .stamp_valid = false, .metadata_valid = true});
    EXPECT_TRUE(missing_stamp.emit);
    EXPECT_EQ(missing_stamp.status, FinalGuestSurfaceStatus::InvalidationLoss);

    const auto invalid_metadata = PlanPpSampledInputObservation(
        {.in_window = true, .stamp_valid = true, .metadata_valid = false});
    EXPECT_TRUE(invalid_metadata.emit);
    EXPECT_EQ(invalid_metadata.status, FinalGuestSurfaceStatus::InvalidationLoss);

    const auto valid = PlanPpSampledInputObservation(
        {.in_window = true, .stamp_valid = true, .metadata_valid = true});
    EXPECT_TRUE(valid.emit);
    EXPECT_EQ(valid.status, FinalGuestSurfaceStatus::Complete);
}

TEST(PpSampledInput, PairedReducerUsesExactOutputAndClassifiesRawBoundaryPerOrdinal) {
    constexpr u32 Width = 32;
    constexpr u32 Height = 32;
    const auto pair = PlanPpSampledInputPairedCapture({
        .enabled = true,
        .width = Width,
        .height = Height,
        .output_format = FinalGuestSurfaceFormat::Bgra8,
        .sampled_format = FinalGuestSurfaceFormat::Rgba16Float,
        .slot_bytes = 16ull << 20,
        .alignment = 256,
    });
    ASSERT_EQ(pair.status, FinalGuestSurfaceStatus::Complete);
    const auto output_plan = PlanFinalGuestSurfaceTiles({
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
        .stage = FinalGuestSurfaceStage::PpSampledInput,
        .logical_width = Width,
        .logical_height = Height,
        .logical_full_fit = true,
        .logical_top_left = true,
        .logical_no_y_flip = true,
    });
    const auto plan = MakePpSampledInputPairedTilePlan(output_plan, pair);
    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(plan.sample_bytes, pair.total_bytes);
    EXPECT_EQ(plan.paired_sampled_offset, pair.sampled_offset);
    EXPECT_EQ(plan.paired_sampled_format, FinalGuestSurfaceFormat::Rgba16Float);

    std::vector<std::byte> a(plan.sample_bytes, std::byte{0x20});
    auto b = a;
    auto c = a;
    for (u32 pixel = 0; pixel < Width * Height / 4; ++pixel) {
        const size_t output = static_cast<size_t>(pixel) * 4;
        b[output] = b[output + 1] = b[output + 2] = std::byte{0xff};
    }
    const auto write_raw = [&](std::vector<std::byte>& bytes, u32 pixel, u16 value) {
        const std::array values{value, value, value, u16{0x3c00}};
        std::memcpy(bytes.data() + pair.sampled_offset +
                        static_cast<size_t>(pixel) * sizeof(values),
                    values.data(), sizeof(values));
    };
    for (u32 pixel = 0; pixel < Width * Height; ++pixel) {
        write_raw(a, pixel, 0x3400);
        write_raw(b, pixel, 0x3400);
        write_raw(c, pixel, 0x3400);
    }
    const auto selector = ParseFinalGuestSurfaceWatchOrdinals("1");
    const auto transport = FinalGuestSurfaceTransport{
        .surface_identity = 1,
        .backing_generation = 1,
        .format = FinalGuestSurfaceFormat::Bgra8,
        .width = Width,
        .height = Height,
    };
    const auto evaluate = [&](const std::vector<std::byte>& middle) {
        FinalGuestSurfaceReducer reducer{FinalGuestSurfaceLagConfig::Defaults(), selector};
        (void)reducer.Observe(10, 1'000'000, transport, plan, a);
        (void)reducer.Observe(11, 1'100'000, transport, plan, middle);
        (void)reducer.Observe(12, 1'200'000, transport, plan, c);
        return reducer.EvaluateCalibratedTriplet({1, 10, 1'000'000, true}, {2, 11, 1'100'000, true},
                                                 {3, 12, 1'200'000, true}, true);
    };

    const auto post_sample = evaluate(b);
    ASSERT_TRUE(post_sample.has_value());
    EXPECT_EQ(post_sample->matched_ordinal_count, 1u);
    EXPECT_EQ(post_sample->post_sample_ordinal_count, 1u);
    EXPECT_EQ(post_sample->post_sample_ordinals[0], 1u);
    EXPECT_EQ(post_sample->pre_or_at_sample_ordinal_count, 0u);
    EXPECT_EQ(post_sample->ambiguous_boundary_ordinal_count, 0u);

    auto raw_return = b;
    for (u32 pixel = 0; pixel < Width * Height / 4; ++pixel) {
        write_raw(raw_return, pixel, 0x3c00);
    }
    const auto pre_sample = evaluate(raw_return);
    ASSERT_TRUE(pre_sample.has_value());
    EXPECT_EQ(pre_sample->pre_or_at_sample_ordinal_count, 1u);
    EXPECT_EQ(pre_sample->pre_or_at_sample_ordinals[0], 1u);
    EXPECT_EQ(pre_sample->post_sample_ordinal_count, 0u);

    const std::string compact = FormatFinalGuestSurfaceCalibratedReport(*pre_sample);
    EXPECT_NE(compact.find(" pre=1"), std::string::npos);
    EXPECT_NE(compact.find(" post="), std::string::npos);
    EXPECT_NE(compact.find(" amb="), std::string::npos);
}

} // namespace
} // namespace Vulkan
