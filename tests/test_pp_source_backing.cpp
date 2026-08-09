// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstddef>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

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
    EXPECT_EQ(PlanPpSourceBackingFootprints(descriptor).status, FinalGuestSurfaceStatus::Complete);
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

} // namespace
} // namespace Vulkan
