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

} // namespace
} // namespace Vulkan
