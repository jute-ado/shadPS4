// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/final_guest_surface_content.h"

namespace {

using Vulkan::FinalGuestSurfaceAspect;
using Vulkan::FinalGuestSurfaceBackingGenerationProvider;
using Vulkan::FinalGuestSurfaceCaptureWindow;
using Vulkan::FinalGuestSurfaceDescriptor;
using Vulkan::FinalGuestSurfaceFormat;
using Vulkan::FinalGuestSurfaceImageType;
using Vulkan::FinalGuestSurfaceLagConfig;
using Vulkan::FinalGuestSurfaceReadbackCompletion;
using Vulkan::FinalGuestSurfaceReadbackSlotPool;
using Vulkan::FinalGuestSurfaceReducer;
using Vulkan::FinalGuestSurfaceReport;
using Vulkan::FinalGuestSurfaceStatus;
using Vulkan::FinalGuestSurfaceTileLimits;
using Vulkan::FinalGuestSurfaceTilePlan;
using Vulkan::FinalGuestSurfaceTransport;

constexpr FinalGuestSurfaceDescriptor Rgba8Surface(u32 width = 1920, u32 height = 1080) {
    return {
        .width = width,
        .height = height,
        .depth = 1,
        .mip_level = 0,
        .mip_levels = 1,
        .base_array_layer = 0,
        .array_layers = 1,
        .samples = 1,
        .type = FinalGuestSurfaceImageType::Color2D,
        .aspect = FinalGuestSurfaceAspect::Color,
        .format = FinalGuestSurfaceFormat::Rgba8,
        .logical_width = width,
        .logical_height = height,
        .logical_full_fit = true,
        .logical_top_left = true,
        .logical_no_y_flip = true,
    };
}

FinalGuestSurfaceTransport Transport(u64 surface, u64 backing) {
    return {
        .surface_identity = surface,
        .backing_generation = backing,
        .format = FinalGuestSurfaceFormat::Rgba8,
        .width = 1920,
        .height = 1080,
    };
}

FinalGuestSurfaceTransport TransportForFormat(FinalGuestSurfaceFormat format) {
    auto transport = Transport(7, 11);
    transport.format = format;
    return transport;
}

TEST(FinalGuestSurfaceContent, PlansCompleteThirtyTwoPixelWindowsOnSixteenPixelStride) {
    const auto first = Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface(1280, 720));
    const auto second = Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface(1280, 720));

    ASSERT_EQ(first.status, FinalGuestSurfaceStatus::Complete);
    ASSERT_EQ(first.tile_count, 79u * 44u);
    EXPECT_EQ(first.logical_columns, 79u);
    EXPECT_EQ(first.logical_rows, 44u);
    EXPECT_EQ(first.sample_bytes, 1280u * 720u * 4u);
    EXPECT_EQ(first, second);
    EXPECT_EQ(first.sample_bytes, second.sample_bytes);
    for (u32 i = 0; i < first.tile_count; ++i) {
        const auto tile = first.TileAt(i);
        EXPECT_EQ(tile.width, 32u);
        EXPECT_EQ(tile.height, 32u);
        EXPECT_EQ(tile.x % 16, 0u);
        EXPECT_EQ(tile.y % 16, 0u);
        EXPECT_LE(tile.x + tile.width, 1280u);
        EXPECT_LE(tile.y + tile.height, 720u);
        EXPECT_EQ(tile.byte_size, tile.width * tile.height * 4u);
    }
    const auto last = first.TileAt(first.tile_count - 1);
    EXPECT_EQ(last.x + last.width, 1280u);
    EXPECT_EQ(last.y + last.height, 720u);
}

TEST(FinalGuestSurfaceContent, CopiesFullSurfaceOnceWithoutDuplicatingLogicalWindows) {
    const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface(1280, 720));

    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    ASSERT_EQ(plan.tile_count, 3476u);
    EXPECT_EQ(plan.sample_bytes, 1280u * 720u * 4u);
    EXPECT_EQ(plan.copy_region_count, 1u);
    EXPECT_LT(plan.sample_bytes, plan.tile_count * 32u * 32u * 4u);
}

TEST(FinalGuestSurfaceContent, MapsPresentedLogicalWindowsExactlyIntoLargerGuestCopy) {
    auto desc = Rgba8Surface(1920, 1080);
    desc.logical_width = 1280;
    desc.logical_height = 720;
    const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(desc);

    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(plan.logical_columns, 79u);
    EXPECT_EQ(plan.logical_rows, 44u);
    EXPECT_EQ(plan.tile_count, 3476u);
    EXPECT_EQ(plan.sample_bytes, 1920u * 1080u * 4u);
    EXPECT_EQ(plan.scale_numerator, 3u);
    EXPECT_EQ(plan.scale_denominator, 2u);
    const auto interior = plan.TileAt(80);
    EXPECT_EQ(interior.x, 24u);
    EXPECT_EQ(interior.y, 24u);
    EXPECT_EQ(interior.width, 48u);
    EXPECT_EQ(interior.height, 48u);
    EXPECT_EQ(plan.TileAt(3475).x + plan.TileAt(3475).width, 1920u);
    EXPECT_EQ(plan.TileAt(3475).y + plan.TileAt(3475).height, 1080u);
    EXPECT_EQ(Vulkan::ValidateFinalGuestSurfaceWatchOrdinals(
                  Vulkan::ParseFinalGuestSurfaceWatchOrdinals("1,3476"), plan.tile_count)
                  .loss,
              0u);
}

TEST(FinalGuestSurfaceContent, RejectsAmbiguousLogicalToGuestMappingsTransactionally) {
    const auto reject = [](FinalGuestSurfaceDescriptor desc) {
        const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(desc);
        EXPECT_EQ(plan.status, FinalGuestSurfaceStatus::Unsupported);
        EXPECT_EQ(plan.loss.logical_mapping, 1u);
        EXPECT_EQ(plan.tile_count, 0u);
        EXPECT_EQ(plan.sample_bytes, 0u);
    };

    auto desc = Rgba8Surface(1920, 1080);
    desc.logical_width = 1280;
    desc.logical_height = 721;
    reject(desc);
    desc.logical_height = 720;
    desc.logical_offset_x = 1;
    reject(desc);
    desc.logical_offset_x = 0;
    desc.logical_no_y_flip = false;
    reject(desc);
    desc.logical_no_y_flip = true;
    desc.logical_full_fit = false;
    reject(desc);
    desc = Rgba8Surface(1280, 720);
    desc.logical_width = 960;
    desc.logical_height = 540;
    reject(desc); // Exact 4/3 scale, but 16 logical pixels do not map to an integer bound.
    desc = Rgba8Surface(1920, 1080);
    desc.logical_width = std::numeric_limits<u32>::max();
    desc.logical_height = std::numeric_limits<u32>::max();
    reject(desc);
}

TEST(FinalGuestSurfaceContent, FullSurfacePlanUsesBoundedSixteenMiBSlotsTransactionally) {
    auto rgba16 = Rgba8Surface(1280, 720);
    rgba16.format = FinalGuestSurfaceFormat::Rgba16;
    const auto supported = Vulkan::PlanFinalGuestSurfaceTiles(rgba16);
    ASSERT_EQ(supported.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(supported.sample_bytes, 1280u * 720u * 8u);
    EXPECT_LE(supported.sample_bytes, FinalGuestSurfaceTileLimits{}.max_bytes);

    const auto too_large = Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface(3840, 2160));
    EXPECT_EQ(too_large.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_EQ(too_large.loss.tile_capacity, 1u);
    EXPECT_EQ(too_large.tile_count, 0u);
    EXPECT_EQ(too_large.sample_bytes, 0u);
}

TEST(FinalGuestSurfaceContent, ClipsNormalizedColumnsForSmallOneDimensionalSurfaces) {
    auto desc = Rgba8Surface(20, 1);
    desc.type = FinalGuestSurfaceImageType::Color1D;
    const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(desc);

    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(plan.tile_count, 1u);
    const auto tile = plan.TileAt(0);
    EXPECT_EQ(tile.y, 0u);
    EXPECT_EQ(tile.height, 1u);
    EXPECT_EQ(tile.width, 20u);
}

TEST(FinalGuestSurfaceContent, RejectsCompressedSurfacesWithoutVisibleAlphaSemantics) {
    auto desc = Rgba8Surface(65, 37);
    desc.format = FinalGuestSurfaceFormat::Block16;
    const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(desc);

    EXPECT_EQ(plan.status, FinalGuestSurfaceStatus::Unsupported);
    EXPECT_EQ(plan.loss.unsupported_format, 1u);
    EXPECT_EQ(plan.tile_count, 0u);
    EXPECT_EQ(plan.sample_bytes, 0u);
}

TEST(FinalGuestSurfaceContent, RejectsMipLayerAspectFormatAndSampleScopeWithoutPartialPlan) {
    const auto reject = [](FinalGuestSurfaceDescriptor desc) {
        const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(desc);
        EXPECT_EQ(plan.status, FinalGuestSurfaceStatus::Unsupported);
        EXPECT_EQ(plan.tile_count, 0u);
        EXPECT_EQ(plan.sample_bytes, 0u);
        EXPECT_TRUE(plan.loss.Any());
        return plan;
    };

    auto desc = Rgba8Surface();
    desc.mip_level = 1;
    EXPECT_EQ(reject(desc).loss.unsupported_mip, 1u);
    desc = Rgba8Surface();
    desc.mip_levels = 2;
    EXPECT_EQ(reject(desc).loss.unsupported_mip, 1u);
    desc = Rgba8Surface();
    desc.base_array_layer = 1;
    EXPECT_EQ(reject(desc).loss.unsupported_layer, 1u);
    desc = Rgba8Surface();
    desc.array_layers = 2;
    EXPECT_EQ(reject(desc).loss.unsupported_layer, 1u);
    desc = Rgba8Surface();
    desc.aspect = FinalGuestSurfaceAspect::Depth;
    EXPECT_EQ(reject(desc).loss.unsupported_aspect, 1u);
    desc = Rgba8Surface();
    desc.format = FinalGuestSurfaceFormat::Unsupported;
    EXPECT_EQ(reject(desc).loss.unsupported_format, 1u);
    desc = Rgba8Surface();
    desc.samples = 2;
    EXPECT_EQ(reject(desc).loss.unsupported_samples, 1u);
    desc = Rgba8Surface();
    desc.type = FinalGuestSurfaceImageType::Color3D;
    EXPECT_EQ(reject(desc).loss.unsupported_type, 1u);
}

TEST(FinalGuestSurfaceContent, AppliesTileAndByteCapsTransactionally) {
    const auto tile_limited = Vulkan::PlanFinalGuestSurfaceTiles(
        Rgba8Surface(1280, 720),
        FinalGuestSurfaceTileLimits{.max_tiles = 3475, .max_bytes = 16u << 20});
    EXPECT_EQ(tile_limited.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_EQ(tile_limited.loss.tile_capacity, 1u);
    EXPECT_EQ(tile_limited.tile_count, 0u);
    EXPECT_EQ(tile_limited.sample_bytes, 0u);

    const auto byte_limited = Vulkan::PlanFinalGuestSurfaceTiles(
        Rgba8Surface(1280, 720), FinalGuestSurfaceTileLimits{
                                     .max_tiles = 4096,
                                     .max_bytes = 1280u * 720u * 4u - 1u,
                                 });
    EXPECT_EQ(byte_limited.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_EQ(byte_limited.loss.byte_capacity, 1u);
    EXPECT_EQ(byte_limited.tile_count, 0u);
    EXPECT_EQ(byte_limited.sample_bytes, 0u);
}

TEST(FinalGuestSurfaceContent, RejectsLogicalWindowLatticeAboveFourThousandNinetySix) {
    const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface(1920, 1080));
    EXPECT_EQ(plan.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_EQ(plan.loss.tile_capacity, 1u);
    EXPECT_EQ(plan.tile_count, 0u);
    EXPECT_EQ(plan.sample_bytes, 0u);
}

TEST(FinalGuestSurfaceContent, WindowIsBoundedAndOverflowSafe) {
    const auto defaults = FinalGuestSurfaceCaptureWindow::Defaults();
    EXPECT_EQ(defaults.frame_count, 1700u);
    EXPECT_LE(defaults.frame_count, FinalGuestSurfaceCaptureWindow::MaxFrameCount);
    EXPECT_TRUE(defaults.Contains(3300));
    EXPECT_TRUE(defaults.Contains(4999));
    EXPECT_FALSE(defaults.Contains(3299));
    EXPECT_FALSE(defaults.Contains(5000));

    const FinalGuestSurfaceCaptureWindow overflow{
        .frame_start = std::numeric_limits<u64>::max() - 1, .frame_count = 8};
    EXPECT_TRUE(overflow.Contains(std::numeric_limits<u64>::max()));
    EXPECT_TRUE(overflow.IsFinal(std::numeric_limits<u64>::max()));
}

TEST(FinalGuestSurfaceContent, ProductionConfigParsingIsBoundedAndDefinesFinalCoverage) {
    const auto read = [](std::string_view name) -> std::optional<std::string_view> {
        if (name == "SHADPS4_FINAL_GUEST_SURFACE_CONTENT") {
            return "1";
        }
        if (name == "SHADPS4_FINAL_GUEST_SURFACE_FRAME_START") {
            return "18446744073709551614";
        }
        if (name == "SHADPS4_FINAL_GUEST_SURFACE_FRAME_COUNT") {
            return "999999";
        }
        if (name == "SHADPS4_FINAL_GUEST_SURFACE_LAG_CADENCE_US") {
            return "200000";
        }
        if (name == "SHADPS4_FINAL_GUEST_SURFACE_LAG_TOLERANCE_US") {
            return "100000";
        }
        if (name == "SHADPS4_FINAL_GUEST_SURFACE_WATCH_ORDINALS") {
            return "1,3476";
        }
        return std::nullopt;
    };

    const auto config = Vulkan::ResolveFinalGuestSurfaceContentConfig(read);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->window.frame_count, FinalGuestSurfaceCaptureWindow::MaxFrameCount);
    EXPECT_EQ(config->lag.cadence_us, 200'000u);
    EXPECT_EQ(config->lag.tolerance_us, 99'999u);
    ASSERT_EQ(config->watch_ordinals.count, 2u);
    EXPECT_EQ(config->watch_ordinals.ordinals[0], 1u);
    EXPECT_EQ(config->watch_ordinals.ordinals[1], 3476u);
    EXPECT_EQ(config->watch_ordinals.loss, 0u);
    EXPECT_TRUE(config->window.IsFinal(std::numeric_limits<u64>::max()));

    const auto disabled = Vulkan::ResolveFinalGuestSurfaceContentConfig(
        [](std::string_view) -> std::optional<std::string_view> { return std::nullopt; });
    EXPECT_FALSE(disabled.has_value());
}

TEST(FinalGuestSurfaceContent, WatchOrdinalSelectorIsStrictBoundedAndTransactional) {
    const auto valid = Vulkan::ParseFinalGuestSurfaceWatchOrdinals("7,1,3476");
    ASSERT_EQ(valid.status, FinalGuestSurfaceStatus::Complete);
    ASSERT_EQ(valid.count, 3u);
    EXPECT_EQ(valid.ordinals[0], 1u);
    EXPECT_EQ(valid.ordinals[1], 7u);
    EXPECT_EQ(valid.ordinals[2], 3476u);
    EXPECT_EQ(valid.loss, 0u);

    for (const std::string_view invalid : {"0", "1, 2", "1,,2", "1,", ",1", "1,1", "4097", "x"}) {
        const auto rejected = Vulkan::ParseFinalGuestSurfaceWatchOrdinals(invalid);
        EXPECT_EQ(rejected.status, FinalGuestSurfaceStatus::Unsupported) << invalid;
        EXPECT_EQ(rejected.count, 0u) << invalid;
        EXPECT_EQ(rejected.loss, 1u) << invalid;
    }

    std::string too_many;
    for (u32 ordinal = 1; ordinal <= Vulkan::FinalGuestSurfaceWatchOrdinals::MaxOrdinals + 1;
         ++ordinal) {
        if (!too_many.empty()) {
            too_many += ',';
        }
        too_many += std::to_string(ordinal);
    }
    const auto capped = Vulkan::ParseFinalGuestSurfaceWatchOrdinals(too_many);
    EXPECT_EQ(capped.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_EQ(capped.count, 0u);
    EXPECT_EQ(capped.loss, 1u);
}

TEST(FinalGuestSurfaceContent, RejectsWatchOrdinalOutsideActualPlanAndClearsHistory) {
    const auto selector = Vulkan::ParseFinalGuestSurfaceWatchOrdinals("3477");
    FinalGuestSurfaceReducer reducer{FinalGuestSurfaceLagConfig::Defaults(), selector};
    const auto larger_plan = Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface(1296, 720));
    const auto route_plan = Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface(1280, 720));
    ASSERT_GT(larger_plan.tile_count, 3476u);
    ASSERT_EQ(route_plan.tile_count, 3476u);
    std::vector<std::byte> a(larger_plan.sample_bytes, std::byte{1});
    std::vector<std::byte> b(larger_plan.sample_bytes, std::byte{2});
    std::vector<std::byte> route(route_plan.sample_bytes, std::byte{1});

    (void)reducer.Observe(1, 1'000'000, Transport(1, 1), larger_plan, a);
    (void)reducer.Observe(2, 1'100'000, Transport(1, 1), larger_plan, b);
    const auto rejected = reducer.Observe(3, 1'200'000, Transport(1, 1), route_plan, route);
    EXPECT_EQ(rejected.status, FinalGuestSurfaceStatus::Unsupported);
    EXPECT_EQ(rejected.selector_status, FinalGuestSurfaceStatus::Unsupported);
    EXPECT_EQ(rejected.selector_loss, 1u);
    EXPECT_FALSE(rejected.exact_aba);
    EXPECT_FALSE(rejected.stable_transport);
    EXPECT_EQ(rejected.aba_tile_ordinal_count, 0u);
    EXPECT_NE(Vulkan::FormatFinalGuestSurfaceReport(rejected).find("selector_loss=1"),
              std::string::npos);

    const auto after_loss = reducer.Observe(4, 1'300'000, Transport(1, 1), larger_plan, a);
    EXPECT_FALSE(after_loss.exact_aba);
    EXPECT_FALSE(after_loss.stable_transport);
}

TEST(FinalGuestSurfaceContent, DisabledAndOutsideWindowPerformZeroPlanningAndStateWork) {
    u32 work{};
    const auto planner = [&] {
        ++work;
        return Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface());
    };
    EXPECT_FALSE(Vulkan::ShouldCaptureFinalGuestSurface(false, {}, 3300));
    EXPECT_FALSE(Vulkan::ShouldCaptureFinalGuestSurface(true, {}, 3299));
    if (Vulkan::ShouldCaptureFinalGuestSurface(false, {}, 3300)) {
        (void)planner();
    }
    if (Vulkan::ShouldCaptureFinalGuestSurface(true, {}, 3299)) {
        (void)planner();
    }
    EXPECT_EQ(work, 0u);
}

TEST(FinalGuestSurfaceContent, ScreenshotCalibrationUsesExactFrameStampAndBoundedRequestOrder) {
    Vulkan::FinalGuestSurfaceFrameDiagnosticStamp stamp;
    stamp.Assign(true, 4572, 84'900'000, 1280, 720);
    const Vulkan::FinalGuestSurfacePresentationMapping identity{
        .guest_width = 1280,
        .guest_height = 720,
        .swapchain_width = 1280,
        .swapchain_height = 720,
        .output_x = 0,
        .output_y = 0,
        .output_width = 1280,
        .output_height = 720,
        .top_left = true,
        .no_y_flip = true,
    };
    Vulkan::FinalGuestSurfaceScreenshotCalibration calibration{true};
    const auto first = calibration.Observe(stamp, identity, 84'900'123);
    EXPECT_TRUE(first.emit);
    EXPECT_EQ(first.request_ordinal, 1u);
    EXPECT_EQ(first.surface_sequence, 4572u);
    EXPECT_EQ(first.surface_process_time_us, 84'900'000u);
    EXPECT_FALSE(first.fallback_time);
    EXPECT_TRUE(first.identity_mapping);
    EXPECT_TRUE(first.exact_scaled_mapping);
    EXPECT_EQ(first.scale_numerator, 1u);
    EXPECT_EQ(first.scale_denominator, 1u);
    EXPECT_EQ(first.mapping_ordinal, 1u);
    EXPECT_TRUE(first.emit_mapping);

    for (u32 request = 2; request <= calibration.MaxRequests; ++request) {
        EXPECT_TRUE(calibration.Observe(stamp, identity, 0).emit);
    }
    const auto overflow = calibration.Observe(stamp, identity, 0);
    EXPECT_TRUE(overflow.emit);
    EXPECT_TRUE(overflow.overflow_marker);
    EXPECT_EQ(overflow.overflow_loss, 1u);
    EXPECT_FALSE(calibration.Observe(stamp, identity, 0).emit);
}

TEST(FinalGuestSurfaceContent, CalibrationReportsExactRationalScaledMappingAndChangesOnce) {
    Vulkan::FinalGuestSurfaceFrameDiagnosticStamp stamp;
    stamp.Assign(true, 4572, 84'900'000, 1920, 1080);
    const Vulkan::FinalGuestSurfacePresentationMapping scaled{
        .guest_width = 1920,
        .guest_height = 1080,
        .swapchain_width = 1280,
        .swapchain_height = 720,
        .output_x = 0,
        .output_y = 0,
        .output_width = 1280,
        .output_height = 720,
        .top_left = true,
        .no_y_flip = true,
    };
    Vulkan::FinalGuestSurfaceScreenshotCalibration calibration{true};
    const auto first = calibration.Observe(stamp, scaled, 0);
    EXPECT_TRUE(first.exact_scaled_mapping);
    EXPECT_FALSE(first.identity_mapping);
    EXPECT_EQ(first.scale_numerator, 3u);
    EXPECT_EQ(first.scale_denominator, 2u);
    EXPECT_EQ(first.mapping_ordinal, 1u);
    EXPECT_TRUE(first.emit_mapping);
    const auto repeated = calibration.Observe(stamp, scaled, 0);
    EXPECT_EQ(repeated.mapping_ordinal, 1u);
    EXPECT_FALSE(repeated.emit_mapping);

    auto changed = scaled;
    changed.output_x = 1;
    const auto changed_report = calibration.Observe(stamp, changed, 0);
    EXPECT_EQ(changed_report.mapping_ordinal, 2u);
    EXPECT_TRUE(changed_report.emit_mapping);
    EXPECT_FALSE(changed_report.exact_scaled_mapping);
}

TEST(FinalGuestSurfaceContent, CalibrationMappingOrdinalsAreBoundedTransactionally) {
    Vulkan::FinalGuestSurfaceFrameDiagnosticStamp stamp;
    stamp.Assign(true, 1, 1, 1280, 720);
    Vulkan::FinalGuestSurfaceScreenshotCalibration calibration{true};
    Vulkan::FinalGuestSurfacePresentationMapping mapping{
        .guest_width = 1280,
        .guest_height = 720,
        .swapchain_width = 1280,
        .swapchain_height = 720,
        .output_width = 1280,
        .output_height = 720,
        .top_left = true,
        .no_y_flip = true,
    };
    for (u32 index = 0; index < calibration.MaxMappingOrdinals; ++index) {
        mapping.output_x = static_cast<s32>(index);
        const auto report = calibration.Observe(stamp, mapping, 0);
        EXPECT_EQ(report.mapping_ordinal, index + 1);
        EXPECT_EQ(report.mapping_loss, 0u);
        EXPECT_TRUE(report.emit_mapping);
    }
    mapping.output_x = static_cast<s32>(calibration.MaxMappingOrdinals);
    const auto overflow = calibration.Observe(stamp, mapping, 0);
    EXPECT_EQ(overflow.mapping_ordinal, 0u);
    EXPECT_EQ(overflow.mapping_loss, 1u);
    EXPECT_FALSE(overflow.emit_mapping);
}

TEST(FinalGuestSurfaceContent, ScreenshotCalibrationCountsOnlySilentAutomationRequests) {
    EXPECT_EQ(Vulkan::FinalGuestSurfaceAutomationCalibrationCount(0, 3), 3u);
    EXPECT_EQ(Vulkan::FinalGuestSurfaceAutomationCalibrationCount(7, 3), 3u);
    EXPECT_EQ(Vulkan::FinalGuestSurfaceAutomationCalibrationCount(7, 0), 0u);
}

TEST(FinalGuestSurfaceContent, CalibrationClearsReusedFramesAndDoesNotClaimScaledOrFlippedMaps) {
    Vulkan::FinalGuestSurfaceFrameDiagnosticStamp stamp;
    stamp.Assign(true, 88, 9'000'000, 1280, 720);
    stamp.Clear();
    const Vulkan::FinalGuestSurfacePresentationMapping scaled{
        .guest_width = 1280,
        .guest_height = 720,
        .swapchain_width = 1920,
        .swapchain_height = 1080,
        .output_x = 0,
        .output_y = 0,
        .output_width = 1920,
        .output_height = 1080,
        .top_left = true,
        .no_y_flip = true,
    };
    Vulkan::FinalGuestSurfaceScreenshotCalibration enabled{true};
    const auto fallback = enabled.Observe(stamp, scaled, 9'100'000);
    EXPECT_EQ(fallback.surface_sequence, 0u);
    EXPECT_EQ(fallback.surface_process_time_us, 9'100'000u);
    EXPECT_TRUE(fallback.fallback_time);
    EXPECT_FALSE(fallback.identity_mapping);

    auto flipped = scaled;
    flipped.swapchain_width = 1280;
    flipped.swapchain_height = 720;
    flipped.output_width = 1280;
    flipped.output_height = 720;
    flipped.no_y_flip = false;
    EXPECT_FALSE(enabled.Observe(stamp, flipped, 9'200'000).identity_mapping);

    Vulkan::FinalGuestSurfaceScreenshotCalibration disabled{false};
    EXPECT_FALSE(disabled.Observe(stamp, scaled, 9'300'000).emit);
    EXPECT_EQ(disabled.ObservedRequests(), 0u);
}

TEST(FinalGuestSurfaceContent, CpuConsumerOwnsEightSlotsUntilExplicitRelease) {
    FinalGuestSurfaceReadbackSlotPool pool;
    std::array<FinalGuestSurfaceReadbackSlotPool::Token,
               FinalGuestSurfaceReadbackSlotPool::MaxSlots>
        tokens{};
    for (auto& token : tokens) {
        const auto acquired = pool.TryAcquire();
        ASSERT_TRUE(acquired);
        token = *acquired;
    }
    EXPECT_FALSE(pool.TryAcquire());
    EXPECT_TRUE(pool.ReleaseAfterCpuConsume(tokens[4]));
    const auto recycled = pool.TryAcquire();
    ASSERT_TRUE(recycled);
    EXPECT_EQ(recycled->slot, tokens[4].slot);
    EXPECT_NE(recycled->generation, tokens[4].generation);
    EXPECT_FALSE(pool.ReleaseAfterCpuConsume(tokens[4]));
    EXPECT_TRUE(pool.ReleaseAfterCpuConsume(*recycled));
}

TEST(FinalGuestSurfaceContent, NoncoherentInvalidationOccursAfterCompletionExactlyOnceWithoutWait) {
    FinalGuestSurfaceReadbackCompletion completion;
    u32 invalidations{};
    const auto invalidate = [&] {
        ++invalidations;
        return true;
    };

    EXPECT_EQ(completion.TryConsume(false, false, invalidate), FinalGuestSurfaceStatus::Pending);
    EXPECT_EQ(invalidations, 0u);
    EXPECT_EQ(completion.TryConsume(true, false, invalidate), FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(invalidations, 1u);
    EXPECT_EQ(completion.TryConsume(true, false, invalidate),
              FinalGuestSurfaceStatus::AlreadyConsumed);
    EXPECT_EQ(invalidations, 1u);

    FinalGuestSurfaceReadbackCompletion failed;
    EXPECT_EQ(failed.TryConsume(true, false, [] { return false; }),
              FinalGuestSurfaceStatus::InvalidationLoss);
}

TEST(FinalGuestSurfaceContent, CoherentConsumptionSkipsInvalidation) {
    FinalGuestSurfaceReadbackCompletion completion;
    u32 invalidations{};
    EXPECT_EQ(completion.TryConsume(true, true,
                                    [&] {
                                        ++invalidations;
                                        return true;
                                    }),
              FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(invalidations, 0u);
}

TEST(FinalGuestSurfaceContent, BackingGenerationIsStablePerBackingAndNeverReused) {
    FinalGuestSurfaceBackingGenerationProvider provider;
    u64 first{};
    u64 second{};
    const u64 first_generation = provider.Assign(first);
    EXPECT_NE(first_generation, 0u);
    EXPECT_EQ(provider.Assign(first), first_generation);
    const u64 second_generation = provider.Assign(second);
    EXPECT_NE(second_generation, 0u);
    EXPECT_NE(second_generation, first_generation);
    EXPECT_EQ(provider.Assign(first), first_generation);
}

TEST(FinalGuestSurfaceContent, LagReducerFindsJitteredHundredMillisecondAba) {
    FinalGuestSurfaceReducer reducer{FinalGuestSurfaceLagConfig::Defaults()};
    const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface(32, 32));
    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    std::vector<std::byte> a(plan.sample_bytes, std::byte{0x11});
    std::vector<std::byte> b(plan.sample_bytes, std::byte{0x22});
    const auto observe = [&](u64 seq, u64 time, const std::vector<std::byte>& bytes) {
        return reducer.Observe(seq, time, Transport(7, 11), plan, bytes);
    };

    const auto first = observe(100, 1'000'000, a);
    EXPECT_FALSE(first.stable_transport);
    EXPECT_FALSE(first.exact_aba);
    (void)observe(101, 1'020'000, a);
    (void)observe(102, 1'098'000, b);
    (void)observe(103, 1'120'000, b);
    (void)observe(104, 1'180'000, b);
    const auto returned = observe(105, 1'201'000, a);
    EXPECT_TRUE(returned.exact_aba);
    EXPECT_TRUE(returned.stable_transport);
    EXPECT_GT(returned.aba_tiles, 0u);
    EXPECT_EQ(returned.loss.gap, 0u);
    EXPECT_EQ(returned.a_sequence, 100u);
    EXPECT_EQ(returned.a_process_time_us, 1'000'000u);
    EXPECT_EQ(returned.b_sequence, 102u);
    EXPECT_EQ(returned.b_process_time_us, 1'098'000u);
    EXPECT_EQ(returned.c_sequence, 105u);
    EXPECT_EQ(returned.c_process_time_us, 1'201'000u);
}

TEST(FinalGuestSurfaceContent, ReportsTrueHistoryEvictionWhenLagEndpointsCannotBeRetained) {
    FinalGuestSurfaceReducer reducer{{.cadence_us = 1'000'000, .tolerance_us = 1}};
    const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface(32, 32));
    std::vector<std::byte> content(plan.sample_bytes, std::byte{0x11});
    Vulkan::FinalGuestSurfaceReport report{};
    for (u32 index = 0; index <= FinalGuestSurfaceReducer::MaxHistory + 1; ++index) {
        report = reducer.Observe(index + 1, 2'000'000 + index * 10,
                                 TransportForFormat(FinalGuestSurfaceFormat::Rgba8), plan, content);
    }
    EXPECT_EQ(report.loss.history, 1u);
    EXPECT_FALSE(report.exact_aba);
    EXPECT_FALSE(report.stable_transport);
}

TEST(FinalGuestSurfaceContent, OutsideToleranceAndTimeReversalResetComparisonState) {
    const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface(32, 32));
    std::vector<std::byte> a(plan.sample_bytes, std::byte{0x11});
    std::vector<std::byte> b(plan.sample_bytes, std::byte{0x22});

    FinalGuestSurfaceReducer outside{FinalGuestSurfaceLagConfig::Defaults()};
    (void)outside.Observe(1, 1'000'000, Transport(1, 1), plan, a);
    (void)outside.Observe(2, 1'070'000, Transport(1, 1), plan, b);
    const auto no_match = outside.Observe(3, 1'200'000, Transport(1, 1), plan, a);
    EXPECT_FALSE(no_match.exact_aba);
    EXPECT_FALSE(no_match.stable_transport);
    EXPECT_EQ(no_match.a_sequence, 0u);
    EXPECT_EQ(no_match.b_sequence, 0u);
    EXPECT_EQ(no_match.c_sequence, 0u);

    FinalGuestSurfaceReducer reversed{FinalGuestSurfaceLagConfig::Defaults()};
    (void)reversed.Observe(10, 2'000'000, Transport(1, 1), plan, a);
    (void)reversed.Observe(11, 2'100'000, Transport(1, 1), plan, b);
    const auto reset = reversed.Observe(12, 2'050'000, Transport(1, 1), plan, a);
    EXPECT_EQ(reset.status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(reset.loss.gap, 1u);
    EXPECT_FALSE(reset.exact_aba);
    EXPECT_FALSE(reset.stable_transport);
    const auto after_reset = reversed.Observe(13, 2'150'000, Transport(1, 1), plan, b);
    EXPECT_FALSE(after_reset.exact_aba);
    EXPECT_FALSE(after_reset.stable_transport);
}

TEST(FinalGuestSurfaceContent, LagReducerSuppressesGapIncompleteAndHistoryLoss) {
    FinalGuestSurfaceReducer reducer{FinalGuestSurfaceLagConfig::Defaults()};
    const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface(32, 32));
    std::vector<std::byte> a(plan.sample_bytes, std::byte{0x11});
    std::vector<std::byte> b(plan.sample_bytes, std::byte{0x22});
    (void)reducer.Observe(10, 1'000'000, Transport(1, 1), plan, a);
    const auto gap = reducer.Observe(12, 1'100'000, Transport(1, 1), plan, b);
    EXPECT_EQ(gap.loss.gap, 1u);
    EXPECT_FALSE(gap.exact_aba);

    auto incomplete = plan;
    incomplete.status = FinalGuestSurfaceStatus::BusyLoss;
    incomplete.loss.busy = 1;
    const auto loss = reducer.Observe(13, 1'200'000, Transport(1, 1), incomplete, {});
    EXPECT_TRUE(loss.loss.Any());
    EXPECT_FALSE(loss.exact_aba);

    const auto reset = reducer.Observe(14, 1'300'000, Transport(1, 1), plan, a);
    EXPECT_FALSE(reset.stable_transport);
    EXPECT_FALSE(reset.exact_aba);
}

TEST(FinalGuestSurfaceContent, StableTransportRequiresEveryIntermediateBackingGeneration) {
    FinalGuestSurfaceReducer reducer{FinalGuestSurfaceLagConfig::Defaults()};
    const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface(32, 32));
    std::vector<std::byte> a(plan.sample_bytes, std::byte{0x11});
    std::vector<std::byte> b(plan.sample_bytes, std::byte{0x22});
    (void)reducer.Observe(20, 2'000'000, Transport(1, 4), plan, a);
    (void)reducer.Observe(21, 2'100'000, Transport(1, 5), plan, b);
    const auto returned = reducer.Observe(22, 2'200'000, Transport(1, 4), plan, a);
    EXPECT_TRUE(returned.exact_aba);
    EXPECT_FALSE(returned.stable_transport);
    EXPECT_EQ(returned.loss.gap, 0u);
}

TEST(FinalGuestSurfaceContent, ReportsLocalizedTileAbaWhenWholeSampleDoesNotReturn) {
    FinalGuestSurfaceReducer reducer{FinalGuestSurfaceLagConfig::Defaults(),
                                     Vulkan::ParseFinalGuestSurfaceWatchOrdinals("1")};
    const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface(80, 32));
    std::vector<std::byte> a(plan.sample_bytes);
    auto b = a;
    auto c = a;
    for (u32 y = 0; y < 32; ++y) {
        for (u32 x = 0; x < 16; ++x) {
            b[(y * 80 + x) * 4] = std::byte{7};
        }
        for (u32 x = 64; x < 80; ++x) {
            c[(y * 80 + x) * 4] = std::byte{9};
        }
    }
    (void)reducer.Observe(100, 1'000'000, TransportForFormat(FinalGuestSurfaceFormat::Rgba8), plan,
                          a);
    (void)reducer.Observe(101, 1'100'000, TransportForFormat(FinalGuestSurfaceFormat::Rgba8), plan,
                          b);
    const auto report = reducer.Observe(
        102, 1'200'000, TransportForFormat(FinalGuestSurfaceFormat::Rgba8), plan, c);

    EXPECT_TRUE(report.exact_aba);
    EXPECT_FALSE(report.whole_sample_aba);
    EXPECT_TRUE(report.stable_transport);
    EXPECT_EQ(report.aba_tiles, 1u);
    EXPECT_EQ(report.unselected_aba_tiles, 0u);
    ASSERT_EQ(report.aba_tile_ordinal_count, 1u);
    EXPECT_EQ(report.aba_tile_ordinals[0], 1u);
    EXPECT_EQ(report.loss.tile_detail, 0u);
}

TEST(FinalGuestSurfaceContent, LogsOnlyTaskSelectedMatchingWindowOrdinals) {
    const auto selector = Vulkan::ParseFinalGuestSurfaceWatchOrdinals("1,2000,3476");
    FinalGuestSurfaceReducer reducer{FinalGuestSurfaceLagConfig::Defaults(), selector};
    const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface(1280, 720));
    const u32 tile_count = plan.tile_count;
    std::vector<std::byte> a(plan.sample_bytes, std::byte{1});
    std::vector<std::byte> b(plan.sample_bytes, std::byte{2});
    (void)reducer.Observe(200, 2'000'000, TransportForFormat(FinalGuestSurfaceFormat::Rgba8), plan,
                          a);
    (void)reducer.Observe(201, 2'100'000, TransportForFormat(FinalGuestSurfaceFormat::Rgba8), plan,
                          b);
    const auto report = reducer.Observe(
        202, 2'200'000, TransportForFormat(FinalGuestSurfaceFormat::Rgba8), plan, a);

    EXPECT_EQ(report.aba_tiles, tile_count);
    EXPECT_EQ(report.aba_tile_ordinal_count, 3u);
    EXPECT_EQ(report.unselected_aba_tiles, tile_count - 3);
    EXPECT_EQ(report.loss.tile_detail, 0u);
    EXPECT_EQ(report.aba_tile_ordinals[0], 1u);
    EXPECT_EQ(report.aba_tile_ordinals[1], 2000u);
    EXPECT_EQ(report.aba_tile_ordinals[2], 3476u);
    const std::string text = Vulkan::FormatFinalGuestSurfaceReport(report);
    EXPECT_NE(text.find("aba_tile_ordinals=1,2000,3476"), std::string::npos);
    EXPECT_NE(text.find("unselected_aba_tiles=3473"), std::string::npos);
    EXPECT_NE(text.find("tile_detail_loss=0"), std::string::npos);
    EXPECT_EQ(text.find("x="), std::string::npos);
    EXPECT_EQ(text.find("y="), std::string::npos);
}

TEST(FinalGuestSurfaceContent, MasksForcedAlphaButRetainsVisibleColorChanges) {
    const auto verify = [](FinalGuestSurfaceFormat format, std::vector<std::byte> a,
                           std::vector<std::byte> alpha_only, std::vector<std::byte> color_change) {
        auto desc = Rgba8Surface(32, 32);
        desc.format = format;
        const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(desc);
        a.resize(plan.sample_bytes);
        alpha_only.resize(plan.sample_bytes);
        color_change.resize(plan.sample_bytes);
        FinalGuestSurfaceReducer alpha_reducer{FinalGuestSurfaceLagConfig::Defaults(),
                                               Vulkan::ParseFinalGuestSurfaceWatchOrdinals("1")};
        (void)alpha_reducer.Observe(300, 3'000'000, TransportForFormat(format), plan, a);
        const auto alpha_changed =
            alpha_reducer.Observe(301, 3'100'000, TransportForFormat(format), plan, alpha_only);
        EXPECT_EQ(alpha_changed.changed_tiles, 0u);
        const auto alpha_return =
            alpha_reducer.Observe(302, 3'200'000, TransportForFormat(format), plan, a);
        EXPECT_FALSE(alpha_return.exact_aba);
        EXPECT_EQ(alpha_return.aba_tiles, 0u);

        FinalGuestSurfaceReducer color_reducer{FinalGuestSurfaceLagConfig::Defaults(),
                                               Vulkan::ParseFinalGuestSurfaceWatchOrdinals("1")};
        (void)color_reducer.Observe(400, 4'000'000, TransportForFormat(format), plan, a);
        EXPECT_EQ(
            color_reducer.Observe(401, 4'100'000, TransportForFormat(format), plan, color_change)
                .changed_tiles,
            1u);
        const auto color_return =
            color_reducer.Observe(402, 4'200'000, TransportForFormat(format), plan, a);
        EXPECT_TRUE(color_return.exact_aba);
        EXPECT_EQ(color_return.aba_tiles, 1u);
    };

    verify(FinalGuestSurfaceFormat::Rgba8, {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{0}},
           {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{0xFF}},
           {std::byte{9}, std::byte{2}, std::byte{3}, std::byte{0}});
    verify(FinalGuestSurfaceFormat::Bgra8, {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{0}},
           {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{0xFF}},
           {std::byte{9}, std::byte{2}, std::byte{3}, std::byte{0}});
    verify(FinalGuestSurfaceFormat::A2R10G10B10,
           {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{0}},
           {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{0xC0}},
           {std::byte{9}, std::byte{2}, std::byte{3}, std::byte{0}});
    verify(FinalGuestSurfaceFormat::A2B10G10R10,
           {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{0}},
           {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{0xC0}},
           {std::byte{9}, std::byte{2}, std::byte{3}, std::byte{0}});
    verify(FinalGuestSurfaceFormat::Rgba16,
           {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}, std::byte{6},
            std::byte{0}, std::byte{0}},
           {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}, std::byte{6},
            std::byte{0xFF}, std::byte{0xFF}},
           {std::byte{9}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}, std::byte{6},
            std::byte{0}, std::byte{0}});
}

TEST(FinalGuestSurfaceContent, ReportsBoundedPrivacySafeSurfaceOrdinalCapacity) {
    FinalGuestSurfaceReducer reducer{FinalGuestSurfaceLagConfig::Defaults()};
    const auto plan = Vulkan::PlanFinalGuestSurfaceTiles(Rgba8Surface(32, 32));
    std::vector<std::byte> content(plan.sample_bytes, std::byte{0x33});
    for (u32 ordinal = 1; ordinal <= FinalGuestSurfaceReducer::MaxSurfaceOrdinals; ++ordinal) {
        const auto report = reducer.Observe(ordinal, 3'000'000 + ordinal * 20'000,
                                            Transport(ordinal, ordinal), plan, content);
        EXPECT_EQ(report.surface_ordinal, ordinal);
        EXPECT_EQ(report.loss.ordinal_capacity, 0u);
    }
    const auto overflow =
        reducer.Observe(FinalGuestSurfaceReducer::MaxSurfaceOrdinals + 1,
                        3'000'000 + (FinalGuestSurfaceReducer::MaxSurfaceOrdinals + 1) * 20'000,
                        Transport(99, 99), plan, content);
    EXPECT_EQ(overflow.surface_ordinal, 0u);
    EXPECT_EQ(overflow.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_EQ(overflow.loss.ordinal_capacity, 1u);
    EXPECT_FALSE(overflow.exact_aba);
}

TEST(FinalGuestSurfaceContent, PrivacySafeReportContainsNoIdentityGenerationDigestOrAddress) {
    Vulkan::FinalGuestSurfaceReport report{
        .sequence = 77,
        .process_time_us = 8'800'000,
        .surface_ordinal = 3,
        .tile_count = 144,
        .changed_tiles = 4,
        .aba_tiles = 2,
        .status = FinalGuestSurfaceStatus::Complete,
        .stable_transport = true,
        .exact_aba = true,
    };
    const std::string text = Vulkan::FormatFinalGuestSurfaceReport(report);
    EXPECT_NE(text.find("sequence=77"), std::string::npos);
    EXPECT_NE(text.find("surface_ordinal=3"), std::string::npos);
    EXPECT_EQ(text.find("identity"), std::string::npos);
    EXPECT_EQ(text.find("generation"), std::string::npos);
    EXPECT_EQ(text.find("digest"), std::string::npos);
    EXPECT_EQ(text.find("address"), std::string::npos);
    EXPECT_EQ(text.find("bytes="), std::string::npos);
}

TEST(FinalGuestSurfaceContent, CompactFrameAndCalibrationLogsStayWithinConservativeBudget) {
    FinalGuestSurfaceReport report{
        .sequence = 4999,
        .process_time_us = 99'999'999,
        .a_sequence = 4993,
        .b_sequence = 4996,
        .c_sequence = 4999,
        .surface_ordinal = 3,
        .tile_count = 3476,
        .changed_tiles = 27,
        .aba_tiles = 2,
        .unselected_aba_tiles = 1,
        .aba_tile_ordinals = {7},
        .aba_tile_ordinal_count = 1,
        .selector_count = 2,
        .selector_status = FinalGuestSurfaceStatus::Complete,
        .status = FinalGuestSurfaceStatus::Complete,
        .stable_transport = true,
        .exact_aba = true,
    };
    const std::string frame = Vulkan::FormatFinalGuestSurfaceCompactReport(report);
    EXPECT_LT(frame.size(), 250u);
    EXPECT_NE(frame.find("q=7"), std::string::npos);
    EXPECT_NE(frame.find("abc=4993/4996/4999"), std::string::npos);

    Vulkan::FinalGuestSurfaceFrameDiagnosticStamp stamp;
    stamp.Assign(true, 4572, 84'900'000, 1920, 1080);
    const Vulkan::FinalGuestSurfacePresentationMapping mapping{
        .guest_width = 1920,
        .guest_height = 1080,
        .swapchain_width = 1280,
        .swapchain_height = 720,
        .output_width = 1280,
        .output_height = 720,
        .top_left = true,
        .no_y_flip = true,
    };
    Vulkan::FinalGuestSurfaceScreenshotCalibration calibration{true};
    const auto first = calibration.Observe(stamp, mapping, 0);
    EXPECT_LT(Vulkan::FormatFinalGuestSurfaceCompactMapping(first).size(), 180u);
    EXPECT_LT(Vulkan::FormatFinalGuestSurfaceCompactCalibration(first).size(), 100u);
    const auto repeated = calibration.Observe(stamp, mapping, 0);
    EXPECT_FALSE(repeated.emit_mapping);
    EXPECT_LT(Vulkan::FormatFinalGuestSurfaceCompactCalibration(repeated).size(), 100u);

    auto worst_frame = report;
    worst_frame.aba_tile_ordinal_count = FinalGuestSurfaceReport::MaxTileDetails;
    worst_frame.aba_tile_ordinals.fill(FinalGuestSurfaceTilePlan::MaxTiles);
    EXPECT_LT(Vulkan::FormatFinalGuestSurfaceCompactReport(worst_frame).size(), 512u);
    constexpr size_t conservative_run_bytes =
        FinalGuestSurfaceCaptureWindow::MaxFrameCount * 512u +
        Vulkan::FinalGuestSurfaceScreenshotCalibration::MaxRequests * 128u +
        Vulkan::FinalGuestSurfaceScreenshotCalibration::MaxMappingOrdinals * 256u + 4096u;
    EXPECT_LT(conservative_run_bytes, 2u << 20);
    EXPECT_LT(conservative_run_bytes, 16u << 20);
}

TEST(FinalGuestSurfaceContent, CompactLossMaskPreservesEveryLossClass) {
    constexpr std::array members{
        &Vulkan::FinalGuestSurfaceLoss::unsupported_type,
        &Vulkan::FinalGuestSurfaceLoss::unsupported_samples,
        &Vulkan::FinalGuestSurfaceLoss::unsupported_mip,
        &Vulkan::FinalGuestSurfaceLoss::unsupported_layer,
        &Vulkan::FinalGuestSurfaceLoss::unsupported_aspect,
        &Vulkan::FinalGuestSurfaceLoss::unsupported_format,
        &Vulkan::FinalGuestSurfaceLoss::invalid_extent,
        &Vulkan::FinalGuestSurfaceLoss::logical_mapping,
        &Vulkan::FinalGuestSurfaceLoss::tile_capacity,
        &Vulkan::FinalGuestSurfaceLoss::byte_capacity,
        &Vulkan::FinalGuestSurfaceLoss::ordinal_capacity,
        &Vulkan::FinalGuestSurfaceLoss::busy,
        &Vulkan::FinalGuestSurfaceLoss::invalidation,
        &Vulkan::FinalGuestSurfaceLoss::gap,
        &Vulkan::FinalGuestSurfaceLoss::history,
        &Vulkan::FinalGuestSurfaceLoss::tile_detail,
    };
    std::array<u32, members.size() + 1> masks{};
    for (u32 index = 0; index < members.size(); ++index) {
        Vulkan::FinalGuestSurfaceLoss loss{};
        loss.*members[index] = 1;
        masks[index] = Vulkan::FinalGuestSurfaceLossMask(loss, 0);
        EXPECT_NE(masks[index], 0u);
    }
    masks.back() = Vulkan::FinalGuestSurfaceLossMask({}, 1);
    for (u32 left = 0; left < masks.size(); ++left) {
        for (u32 right = left + 1; right < masks.size(); ++right) {
            EXPECT_NE(masks[left], masks[right]);
        }
    }
}

} // namespace
