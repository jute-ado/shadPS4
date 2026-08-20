// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/texture_cache/host_image_scaling.h"

using namespace VideoCore;

TEST(HostImageScaling, PlansExactTwoTimesExtentForEligibleTargets) {
    const auto color = PlanHostImageScale({.width = 1920,
                                           .height = 1080,
                                           .display_width = 1280,
                                           .display_height = 720,
                                           .scale_percent = 200,
                                           .is_render_target = true});
    EXPECT_EQ(color.disposition, HostImageScaleDisposition::Scaled);
    EXPECT_EQ(color.host_width, 3840u);
    EXPECT_EQ(color.host_height, 2160u);
    EXPECT_EQ(color.numerator, 2u);
    EXPECT_EQ(color.denominator, 1u);
    EXPECT_EQ(ScaleHostCoordinate(960u, color), 1920u);
    EXPECT_FLOAT_EQ(ScaleHostCoordinate(540.5f, color), 1081.0f);

    const auto depth = PlanHostImageScale({.width = 1920,
                                           .height = 1080,
                                           .display_width = 1280,
                                           .display_height = 720,
                                           .scale_percent = 200,
                                           .is_depth_target = true});
    EXPECT_EQ(depth.disposition, HostImageScaleDisposition::Scaled);
}

TEST(HostImageScaling, IdentityIsByteForByteCompatible) {
    const auto plan = PlanHostImageScale({.width = 1920,
                                          .height = 1080,
                                          .scale_percent = 100,
                                          .levels = 12,
                                          .is_block_compressed = true});
    EXPECT_EQ(plan.disposition, HostImageScaleDisposition::Identity);
    EXPECT_EQ(plan.host_width, 1920u);
    EXPECT_EQ(plan.host_height, 1080u);
    EXPECT_EQ(ScaleHostCoordinate(37u, plan), 37u);
}

TEST(HostImageScaling, RejectsPartialOrUnsafeScalingTransactionally) {
    const auto reject = [](HostImageScaleDescriptor desc, HostImageScaleDisposition disposition) {
        const auto plan = PlanHostImageScale(desc);
        EXPECT_EQ(plan.disposition, disposition);
        EXPECT_EQ(plan.host_width, desc.width);
        EXPECT_EQ(plan.host_height, desc.height);
        EXPECT_FALSE(plan.IsScaled());
    };

    HostImageScaleDescriptor base{.width = 1920,
                                  .height = 1080,
                                  .display_width = 1280,
                                  .display_height = 720,
                                  .scale_percent = 200,
                                  .is_render_target = true};
    auto unsupported_scale = base;
    unsupported_scale.scale_percent = 150;
    reject(unsupported_scale, HostImageScaleDisposition::UnsupportedScale);
    auto sampled_only = base;
    sampled_only.is_render_target = false;
    reject(sampled_only, HostImageScaleDisposition::UnsupportedImage);
    auto three_dimensional = base;
    three_dimensional.is_2d = false;
    reject(three_dimensional, HostImageScaleDisposition::UnsupportedImage);
    auto compressed = base;
    compressed.is_block_compressed = true;
    reject(compressed, HostImageScaleDisposition::UnsupportedImage);
    auto mipmapped = base;
    mipmapped.levels = 2;
    reject(mipmapped, HostImageScaleDisposition::UnsupportedImage);
    auto multisampled = base;
    multisampled.samples = 4;
    reject(multisampled, HostImageScaleDisposition::UnsupportedImage);
    auto cannot_blit = base;
    cannot_blit.supports_blit = false;
    reject(cannot_blit, HostImageScaleDisposition::UnsupportedImage);
    auto shadow_map = base;
    shadow_map.width = 2048;
    shadow_map.height = 2048;
    reject(shadow_map, HostImageScaleDisposition::UnsupportedImage);
    auto missing_display = base;
    missing_display.display_width = 0;
    reject(missing_display, HostImageScaleDisposition::UnsupportedImage);
    auto overflow = base;
    overflow.width = std::numeric_limits<std::uint32_t>::max();
    reject(overflow, HostImageScaleDisposition::Overflow);
}

TEST(HostImageScaling, NativeFallbackIsScopedToTheExactImageBinding) {
    EXPECT_FALSE(ShouldForceNativeHostImageAccess(false, false, false));
    EXPECT_TRUE(ShouldForceNativeHostImageAccess(true, false, false));
    EXPECT_TRUE(ShouldForceNativeHostImageAccess(false, true, false));
    EXPECT_TRUE(ShouldForceNativeHostImageAccess(false, false, true));
    EXPECT_TRUE(ShouldForceNativeHostImageAccess(true, true, true));
}

TEST(HostImageScaling, MapsHostFragmentCoordinatesBackToGuestPixelSpace) {
    EXPECT_FLOAT_EQ(HostToGuestFragmentCoordinate(0.5f, 1), 0.5f);
    EXPECT_FLOAT_EQ(HostToGuestFragmentCoordinate(0.5f, 2), 0.25f);
    EXPECT_FLOAT_EQ(HostToGuestFragmentCoordinate(1.5f, 2), 0.75f);
    EXPECT_FLOAT_EQ(HostToGuestFragmentCoordinate(3.5f, 2), 1.75f);
}

TEST(HostImageScaling, ResolvesAllAttachmentsAsOneScaleTransaction) {
    const std::array all_eligible{true, true, true};
    EXPECT_EQ(ResolveHostAttachmentScale(200, all_eligible), 2u);

    const std::array one_ineligible{true, false, true};
    EXPECT_EQ(ResolveHostAttachmentScale(200, one_ineligible), 1u);
    EXPECT_EQ(ResolveHostAttachmentScale(100, all_eligible), 1u);

    const std::array<bool, 0> no_attachments{};
    EXPECT_EQ(ResolveHostAttachmentScale(200, no_attachments), 1u);
}
