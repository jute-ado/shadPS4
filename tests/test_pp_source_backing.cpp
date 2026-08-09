// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstddef>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/host_passes/pp_sampled_input.h"
#include "video_core/renderer_vulkan/host_passes/pp_source_backing.h"

namespace Vulkan {
namespace {

constexpr auto Watch(std::string_view text) {
    return ParseFinalGuestSurfaceWatchOrdinals(text);
}

PpSourceBackingFootprintDescriptor RouteDescriptor(
    FinalGuestSurfaceWatchOrdinals selector = Watch("1024")) {
    return {
        .enabled = true,
        .in_window = true,
        .pp_draw_encoded = true,
        .fsr_bypassed = true,
        .source_width = 1920,
        .source_height = 1080,
        .logical_width = 1280,
        .logical_height = 720,
        .source_format = FinalGuestSurfaceFormat::Bgra8,
        .samples = 1,
        .resolved_mip_count = 1,
        .resolved_layer_count = 1,
        .bound_mip_count = 1,
        .bound_layer_count = 1,
        .logical_full_fit = true,
        .logical_top_left = true,
        .logical_no_y_flip = true,
        .buffer_alignment = 16,
        .max_regions = 32,
        .max_bytes = 1u << 20,
        .selector = selector,
    };
}

void FillRegion(std::vector<std::byte>& bytes, const PpSourceBackingRegion& region, std::byte value,
                std::byte alpha = std::byte{0xff}) {
    for (u32 y = 0; y < region.height; ++y) {
        for (u32 x = 0; x < region.width; ++x) {
            const size_t offset =
                region.buffer_offset + (static_cast<size_t>(y) * region.width + x) * 4;
            bytes[offset + 0] = value;
            bytes[offset + 1] = value;
            bytes[offset + 2] = value;
            bytes[offset + 3] = alpha;
        }
    }
}

TEST(PpSourceBacking, DisabledOrOutsideWindowDoesNoWork) {
    auto descriptor = RouteDescriptor();
    descriptor.enabled = false;
    auto plan = PlanPpSourceBackingFootprints(descriptor);
    EXPECT_EQ(plan.status, FinalGuestSurfaceStatus::AlreadyConsumed);
    EXPECT_EQ(plan.region_count, 0u);
    EXPECT_EQ(plan.copy_region_count, 0u);
    EXPECT_EQ(plan.buffer_bytes, 0u);
    EXPECT_FALSE(plan.allocate_snapshot);
    EXPECT_FALSE(plan.cpu_wait);
    EXPECT_FALSE(plan.finish);

    descriptor.enabled = true;
    descriptor.in_window = false;
    plan = PlanPpSourceBackingFootprints(descriptor);
    EXPECT_EQ(plan.status, FinalGuestSurfaceStatus::AlreadyConsumed);
    EXPECT_EQ(plan.copy_region_count, 0u);
}

TEST(PpSourceBacking, MapsLogicalWindowToExactLinearClampSourceFootprint) {
    const auto plan = PlanPpSourceBackingFootprints(RouteDescriptor());
    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    ASSERT_EQ(plan.region_count, 1u);
    const auto& region = plan.regions[0];
    EXPECT_EQ(region.logical_ordinal, 1024u);
    EXPECT_EQ(region.x, 1800u);
    EXPECT_EQ(region.y, 288u);
    EXPECT_EQ(region.width, 48u);
    EXPECT_EQ(region.height, 48u);
    EXPECT_EQ(region.byte_size, 48u * 48u * 4u);
    EXPECT_EQ(region.buffer_offset % 16u, 0u);
    EXPECT_EQ(plan.copy_region_count, 1u);
    EXPECT_EQ(plan.image_barrier_count, 2u);
    EXPECT_TRUE(plan.pp_draw_precedes_copy);
    EXPECT_TRUE(plan.restores_shader_read);
    EXPECT_TRUE(plan.callback_payload_is_scalar_only);
    EXPECT_FALSE(plan.callback_retains_frame);
    EXPECT_FALSE(plan.callback_retains_image);
    EXPECT_FALSE(plan.callback_retains_vk_image);
    EXPECT_FALSE(plan.cpu_wait);
    EXPECT_FALSE(plan.finish);
}

TEST(PpSourceBacking, SelectedFootprintsAreSortedPackedAndBounded) {
    auto descriptor = RouteDescriptor(Watch("390,1024,1299,2031,2504"));
    const auto plan = PlanPpSourceBackingFootprints(descriptor);
    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    ASSERT_EQ(plan.region_count, 5u);
    for (u32 index = 0; index < plan.region_count; ++index) {
        EXPECT_EQ(plan.regions[index].logical_ordinal, descriptor.selector.ordinals[index]);
        EXPECT_EQ(plan.regions[index].buffer_offset % descriptor.buffer_alignment, 0u);
        if (index != 0) {
            EXPECT_GE(plan.regions[index].buffer_offset,
                      plan.regions[index - 1].buffer_offset + plan.regions[index - 1].byte_size);
        }
    }
    EXPECT_LE(plan.buffer_bytes, descriptor.max_bytes);

    descriptor.max_regions = 4;
    EXPECT_EQ(PlanPpSourceBackingFootprints(descriptor).status,
              FinalGuestSurfaceStatus::CapacityLoss);
    descriptor.max_regions = 32;
    descriptor.max_bytes = 1024;
    EXPECT_EQ(PlanPpSourceBackingFootprints(descriptor).status,
              FinalGuestSurfaceStatus::CapacityLoss);
}

TEST(PpSourceBacking, InvalidViewFormatScaleOrOrderingFailsClosed) {
    auto descriptor = RouteDescriptor();
    descriptor.resolved_base_mip = 1;
    EXPECT_EQ(PlanPpSourceBackingFootprints(descriptor).status,
              FinalGuestSurfaceStatus::InvalidationLoss);
    descriptor.bound_base_mip = 1;
    EXPECT_EQ(PlanPpSourceBackingFootprints(descriptor).status,
              FinalGuestSurfaceStatus::Unsupported)
        << "Phase A must not apply base-level extent math to a nonzero bound mip";
    descriptor.samples = 2;
    EXPECT_EQ(PlanPpSourceBackingFootprints(descriptor).status,
              FinalGuestSurfaceStatus::Unsupported);
    descriptor.samples = 1;
    descriptor.source_format = FinalGuestSurfaceFormat::Rgba16Float;
    EXPECT_EQ(PlanPpSourceBackingFootprints(descriptor).status,
              FinalGuestSurfaceStatus::Unsupported);
    descriptor.source_format = FinalGuestSurfaceFormat::Bgra8;
    descriptor.logical_width = 1279;
    EXPECT_EQ(PlanPpSourceBackingFootprints(descriptor).status,
              FinalGuestSurfaceStatus::Unsupported);
    descriptor.logical_width = 1280;
    descriptor.pp_draw_encoded = false;
    EXPECT_EQ(PlanPpSourceBackingFootprints(descriptor).status,
              FinalGuestSurfaceStatus::InvalidationLoss);
}

TEST(PpSourceBacking, UpscaleWithoutBilinearHaloFailsClosed) {
    auto descriptor = RouteDescriptor();
    descriptor.source_width = 640;
    descriptor.source_height = 360;
    EXPECT_EQ(PlanPpSourceBackingFootprints(descriptor).status,
              FinalGuestSurfaceStatus::Unsupported)
        << "Upscaling needs a bilinear halo and must fail closed in the exact-footprint probe";
}

TEST(PpSourceBacking, ExactAbaStableAndChangingBackingAreDistinct) {
    const auto plan = PlanPpSourceBackingFootprints(RouteDescriptor());
    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    std::vector<std::byte> a(plan.buffer_bytes);
    std::vector<std::byte> b(plan.buffer_bytes);
    std::vector<std::byte> c(plan.buffer_bytes);
    FillRegion(a, plan.regions[0], std::byte{0x20});
    FillRegion(b, plan.regions[0], std::byte{0x80});
    FillRegion(c, plan.regions[0], std::byte{0x20});
    auto result = ClassifyPpSourceBackingTriplet(plan, a, b, c);
    ASSERT_EQ(result.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(result.aba_count, 1u);
    EXPECT_EQ(result.aba_ordinals[0], 1024u);
    EXPECT_EQ(result.stable_count, 0u);
    EXPECT_EQ(result.ambiguous_count, 0u);

    b = a;
    result = ClassifyPpSourceBackingTriplet(plan, a, b, c);
    EXPECT_EQ(result.aba_count, 0u);
    EXPECT_EQ(result.stable_count, 1u);
    EXPECT_EQ(result.stable_ordinals[0], 1024u);

    FillRegion(c, plan.regions[0], std::byte{0x21});
    result = ClassifyPpSourceBackingTriplet(plan, a, b, c);
    EXPECT_EQ(result.ambiguous_count, 1u);
    EXPECT_EQ(result.ambiguous_ordinals[0], 1024u);
}

TEST(PpSourceBacking, ForcedAlphaIsIgnoredButVisibleChannelsAreNot) {
    const auto plan = PlanPpSourceBackingFootprints(RouteDescriptor());
    std::vector<std::byte> a(plan.buffer_bytes);
    FillRegion(a, plan.regions[0], std::byte{0x30}, std::byte{0x00});
    auto b = a;
    auto c = a;
    FillRegion(b, plan.regions[0], std::byte{0x30}, std::byte{0xff});
    auto result = ClassifyPpSourceBackingTriplet(plan, a, b, c);
    EXPECT_EQ(result.stable_count, 1u);
    b[plan.regions[0].buffer_offset] = std::byte{0x31};
    result = ClassifyPpSourceBackingTriplet(plan, a, b, c);
    EXPECT_EQ(result.aba_count, 1u);
}

TEST(PpSourceBacking, RingRotationIsAllowedButEveryCaptureMustMatchItsOwnGeneration) {
    const auto a = ValidatePpSourceBackingObservation({
        .logical_generation = 9,
        .expected_image_uid = 101,
        .expected_backing_generation = 11,
        .captured_image_uid = 101,
        .captured_backing_generation = 11,
    });
    const auto b = ValidatePpSourceBackingObservation({
        .logical_generation = 9,
        .expected_image_uid = 102,
        .expected_backing_generation = 12,
        .captured_image_uid = 102,
        .captured_backing_generation = 12,
    });
    const auto c = ValidatePpSourceBackingObservation({
        .logical_generation = 9,
        .expected_image_uid = 103,
        .expected_backing_generation = 13,
        .captured_image_uid = 103,
        .captured_backing_generation = 13,
    });
    EXPECT_TRUE(PpSourceBackingTripletTransportCompatible(a, b, c));

    auto stale = c;
    stale.captured_backing_generation = 12;
    stale = ValidatePpSourceBackingObservation(stale);
    EXPECT_EQ(stale.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_FALSE(PpSourceBackingTripletTransportCompatible(a, b, stale));

    auto changed = c;
    changed.logical_generation = 10;
    changed = ValidatePpSourceBackingObservation(changed);
    EXPECT_FALSE(PpSourceBackingTripletTransportCompatible(a, b, changed));
}

TEST(PpSourceBacking, MissingBytesAndInvalidTransportFailClosedWithoutDetails) {
    const auto plan = PlanPpSourceBackingFootprints(RouteDescriptor());
    std::vector<std::byte> bytes(plan.buffer_bytes);
    const auto short_bytes = std::span<const std::byte>{bytes}.first(bytes.size() - 1);
    const auto result = ClassifyPpSourceBackingTriplet(plan, bytes, bytes, short_bytes);
    EXPECT_EQ(result.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(result.aba_count, 0u);
    EXPECT_EQ(result.stable_count, 0u);
    EXPECT_EQ(result.ambiguous_count, 0u);
}

TEST(PpSourceBacking, PrivacyFormatterContainsOnlyOrdinalsCountsAndStatus) {
    PpSourceBackingTripletReport report{
        .request_ordinal = 193,
        .a_sequence = 4262,
        .b_sequence = 4268,
        .c_sequence = 4273,
        .aba_ordinals = {1024},
        .aba_count = 1,
        .status = FinalGuestSurfaceStatus::Complete,
    };
    const std::string line = FormatPpSourceBackingTripletReport(report);
    EXPECT_NE(line.find("PPSB q=193 abc=4262/4268/4273"), std::string::npos);
    EXPECT_NE(line.find("aba=1024"), std::string::npos);
    EXPECT_EQ(line.find("0x"), std::string::npos);
    EXPECT_EQ(line.find("uid"), std::string::npos);
    EXPECT_EQ(line.find("generation"), std::string::npos);
    EXPECT_LT(line.size(), 256u);
}

TEST(PpSourceBacking, AttachesCompactBackingPlaneToExistingOutputAndRawSlot) {
    constexpr u32 Width = 32;
    constexpr u32 Height = 32;
    const auto output = PlanFinalGuestSurfaceTiles({
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
    const auto pair = PlanPpSampledInputPairedCapture({
        .enabled = true,
        .width = Width,
        .height = Height,
        .output_format = FinalGuestSurfaceFormat::Bgra8,
        .sampled_format = FinalGuestSurfaceFormat::Rgba16Float,
        .slot_bytes = 16ull << 20,
        .alignment = 256,
    });
    const auto backing = PlanPpSourceBackingFootprints(PpSourceBackingFootprintDescriptor{
        .enabled = true,
        .in_window = true,
        .pp_draw_encoded = true,
        .fsr_bypassed = true,
        .source_width = Width,
        .source_height = Height,
        .logical_width = Width,
        .logical_height = Height,
        .source_format = FinalGuestSurfaceFormat::Bgra8,
        .samples = 1,
        .resolved_mip_count = 1,
        .resolved_layer_count = 1,
        .bound_mip_count = 1,
        .bound_layer_count = 1,
        .logical_full_fit = true,
        .logical_top_left = true,
        .logical_no_y_flip = true,
        .buffer_alignment = 16,
        .max_regions = 32,
        .max_bytes = 1u << 20,
        .selector = Watch("1"),
    });
    ASSERT_EQ(backing.status, FinalGuestSurfaceStatus::Complete);
    const auto plan = MakePpSampledInputSourceBackingTilePlan(
        MakePpSampledInputPairedTilePlan(output, pair), backing, 16ull << 20, 16);
    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(plan.copy_region_count, 3u);
    EXPECT_EQ(plan.paired_backing_format, FinalGuestSurfaceFormat::Bgra8);
    EXPECT_EQ(plan.paired_backing_region_count, 1u);
    EXPECT_EQ(plan.paired_backing_regions[0].logical_ordinal, 1u);
    EXPECT_GE(plan.paired_backing_offset, pair.total_bytes);
    EXPECT_EQ(plan.paired_backing_bytes, backing.buffer_bytes);
    EXPECT_EQ(plan.sample_bytes, plan.paired_backing_offset + backing.buffer_bytes);
}

TEST(PpSourceBacking, CalibratedOutputReturnClassifiesExactBackingPerOrdinal) {
    constexpr u32 Width = 32;
    constexpr u32 Height = 32;
    const auto output = PlanFinalGuestSurfaceTiles({
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
    const auto pair = PlanPpSampledInputPairedCapture({
        .enabled = true,
        .width = Width,
        .height = Height,
        .output_format = FinalGuestSurfaceFormat::Bgra8,
        .sampled_format = FinalGuestSurfaceFormat::Rgba16Float,
        .slot_bytes = 16ull << 20,
        .alignment = 256,
    });
    const auto backing = PlanPpSourceBackingFootprints(PpSourceBackingFootprintDescriptor{
        .enabled = true,
        .in_window = true,
        .pp_draw_encoded = true,
        .fsr_bypassed = true,
        .source_width = Width,
        .source_height = Height,
        .logical_width = Width,
        .logical_height = Height,
        .source_format = FinalGuestSurfaceFormat::Bgra8,
        .samples = 1,
        .resolved_mip_count = 1,
        .resolved_layer_count = 1,
        .bound_mip_count = 1,
        .bound_layer_count = 1,
        .logical_full_fit = true,
        .logical_top_left = true,
        .logical_no_y_flip = true,
        .buffer_alignment = 16,
        .max_regions = 32,
        .max_bytes = 1u << 20,
        .selector = Watch("1"),
    });
    const auto plan = MakePpSampledInputSourceBackingTilePlan(
        MakePpSampledInputPairedTilePlan(output, pair), backing, 16ull << 20, 16);
    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);

    std::vector<std::byte> a(plan.sample_bytes, std::byte{0x20});
    auto b = a;
    auto c = a;
    for (u32 pixel = 0; pixel < Width * Height / 4; ++pixel) {
        const size_t output_offset = static_cast<size_t>(pixel) * 4;
        b[output_offset] = b[output_offset + 1] = b[output_offset + 2] = std::byte{0xff};
    }
    const size_t backing_offset = plan.paired_backing_offset;
    b[backing_offset] = std::byte{0x80};

    const auto selector = Watch("1");
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

    const auto aba = evaluate(b, c);
    ASSERT_TRUE(aba.has_value());
    EXPECT_EQ(aba->matched_ordinal_count, 1u);
    EXPECT_EQ(aba->backing_aba_ordinal_count, 1u);
    EXPECT_EQ(aba->backing_aba_ordinals[0], 1u);
    EXPECT_EQ(aba->backing_stable_ordinal_count, 0u);
    EXPECT_EQ(aba->backing_ambiguous_ordinal_count, 0u);

    auto stable_middle = b;
    stable_middle[backing_offset] = a[backing_offset];
    const auto stable = evaluate(stable_middle, c);
    ASSERT_TRUE(stable.has_value());
    EXPECT_EQ(stable->backing_stable_ordinal_count, 1u);
    EXPECT_EQ(stable->backing_stable_ordinals[0], 1u);

    auto changed = c;
    changed[backing_offset] = std::byte{0x21};
    const auto ambiguous = evaluate(b, changed);
    ASSERT_TRUE(ambiguous.has_value());
    EXPECT_EQ(ambiguous->backing_ambiguous_ordinal_count, 1u);
    EXPECT_EQ(ambiguous->backing_ambiguous_ordinals[0], 1u);

    const std::string compact = FormatFinalGuestSurfaceCalibratedReport(*aba);
    EXPECT_NE(compact.find(" ba=1"), std::string::npos);
    EXPECT_NE(compact.find(" bs="), std::string::npos);
    EXPECT_NE(compact.find(" bx="), std::string::npos);
    EXPECT_LT(compact.size(), 320u);
}

TEST(PpSourceBacking, ProductionHandoffSnapshotsAfterDrawAndPacksOnPresent) {
    const auto backing = PlanPpSourceBackingFootprints(RouteDescriptor());
    const auto handoff = PlanPpSourceBackingHandoff({
        .enabled = true,
        .frame_is_new = true,
        .metadata_valid = true,
        .snapshot_buffer_available = true,
        .backing = backing,
    });
    ASSERT_EQ(handoff.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_TRUE(handoff.pp_draw_precedes_source_transition);
    EXPECT_EQ(handoff.draw_source_image_barriers, 2u);
    EXPECT_EQ(handoff.draw_image_to_snapshot_regions, backing.region_count);
    EXPECT_EQ(handoff.draw_snapshot_buffer_barriers, 2u);
    EXPECT_EQ(handoff.present_snapshot_to_readback_copies, 1u);
    EXPECT_TRUE(handoff.present_wait_includes_transfer);
    EXPECT_TRUE(handoff.callback_payload_is_scalar_only);
    EXPECT_FALSE(handoff.cpu_wait);
    EXPECT_FALSE(handoff.finish);
    EXPECT_FALSE(handoff.callback_retains_frame);
    EXPECT_FALSE(handoff.callback_retains_image);
    EXPECT_FALSE(handoff.callback_retains_vk_image);

    EXPECT_FALSE(PlanPpSourceBackingHandoff({
                                                .enabled = true,
                                                .frame_is_new = false,
                                                .metadata_valid = true,
                                                .snapshot_buffer_available = true,
                                                .backing = backing,
                                            })
                     .copy);
    EXPECT_EQ(PlanPpSourceBackingHandoff({
                                             .enabled = true,
                                             .frame_is_new = true,
                                             .metadata_valid = false,
                                             .snapshot_buffer_available = true,
                                             .backing = backing,
                                         })
                  .status,
              FinalGuestSurfaceStatus::InvalidationLoss);
}

} // namespace
} // namespace Vulkan
