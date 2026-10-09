// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <vector>

#include <gtest/gtest.h>

#include "shader_recompiler/ir/passes/msaa_image_argument_policy.h"
#include "video_core/texture_cache/image_binding_policy.h"
#include "video_core/texture_cache/image_usage_policy.h"

namespace VideoCore {
namespace {

constexpr vk::ImageUsageFlags RequiredSampledUsage =
    vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
    vk::ImageUsageFlagBits::eSampled;
constexpr vk::ImageUsageFlags PreferredUsage =
    RequiredSampledUsage | vk::ImageUsageFlagBits::eStorage;

TEST(ImageUsagePolicy, RetainsPreferredUsageWhenTheDeviceSupportsIt) {
    std::vector<vk::ImageUsageFlags> queries;

    const auto result = SelectSupportedImageUsage(
        true, PreferredUsage, [&](vk::ImageUsageFlags usage) {
            queries.push_back(usage);
            return true;
        });

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->usage, PreferredUsage);
    EXPECT_FALSE(result->dropped_optional_storage);
    EXPECT_EQ(queries, std::vector<vk::ImageUsageFlags>{PreferredUsage});
}

TEST(ImageUsagePolicy, DropsOptionalStorageForCompressedImagesWhenRequiredUsageIsSupported) {
    std::vector<vk::ImageUsageFlags> queries;

    const auto result = SelectSupportedImageUsage(
        true, PreferredUsage, [&](vk::ImageUsageFlags usage) {
            queries.push_back(usage);
            return usage == RequiredSampledUsage;
        });

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->usage, RequiredSampledUsage);
    EXPECT_TRUE(result->dropped_optional_storage);
    EXPECT_EQ(queries,
              (std::vector<vk::ImageUsageFlags>{PreferredUsage, RequiredSampledUsage}));
}

TEST(ImageUsagePolicy, DoesNotDropStorageFromUncompressedImages) {
    std::vector<vk::ImageUsageFlags> queries;

    const auto result = SelectSupportedImageUsage(
        false, PreferredUsage, [&](vk::ImageUsageFlags usage) {
            queries.push_back(usage);
            return usage == RequiredSampledUsage;
        });

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(queries, std::vector<vk::ImageUsageFlags>{PreferredUsage});
}

TEST(ImageUsagePolicy, FailsClosedWhenEvenRequiredCompressedUsageIsUnsupported) {
    std::vector<vk::ImageUsageFlags> queries;

    const auto result = SelectSupportedImageUsage(
        true, PreferredUsage, [&](vk::ImageUsageFlags usage) {
            queries.push_back(usage);
            return false;
        });

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(queries,
              (std::vector<vk::ImageUsageFlags>{PreferredUsage, RequiredSampledUsage}));
}

TEST(ImageUsagePolicy, RejectsStorageBindingsWithoutAllUnformattedStorageFeatures) {
    constexpr auto all_storage_features =
        vk::FormatFeatureFlagBits2::eStorageImage |
        vk::FormatFeatureFlagBits2::eStorageReadWithoutFormat |
        vk::FormatFeatureFlagBits2::eStorageWriteWithoutFormat;

    EXPECT_TRUE(IsImageBindingFormatUsable(false, {}));
    EXPECT_TRUE(IsImageBindingFormatUsable(true, all_storage_features));
    EXPECT_FALSE(IsImageBindingFormatUsable(
        true, vk::FormatFeatureFlagBits2::eSampledImage));
    EXPECT_FALSE(IsImageBindingFormatUsable(
        true, vk::FormatFeatureFlagBits2::eStorageImage |
                  vk::FormatFeatureFlagBits2::eStorageReadWithoutFormat));
}

constexpr ImageBindingLimits TestImageLimits{
    .max_dimension_1d = 16384,
    .max_dimension_2d = 16384,
    .max_dimension_3d = 2048,
    .max_dimension_cube = 16384,
    .max_array_layers = 2048,
    .max_guest_resource_size = 8ULL * 1024 * 1024 * 1024,
};

TEST(ImageBindingPolicy, AcceptsValidImageKindsAtDeviceLimits) {
    EXPECT_TRUE(IsImageBindingExtentUsable(
        ImageBindingKind::OneDimensional, {16384, 1, 1, 2048}, TestImageLimits));
    EXPECT_TRUE(IsImageBindingExtentUsable(
        ImageBindingKind::TwoDimensional, {16384, 16384, 1, 2048}, TestImageLimits));
    EXPECT_TRUE(IsImageBindingExtentUsable(
        ImageBindingKind::ThreeDimensional, {2048, 2048, 2048, 1}, TestImageLimits));
    EXPECT_TRUE(IsImageBindingExtentUsable(
        ImageBindingKind::Cube, {16384, 16384, 1, 2046}, TestImageLimits));
}

TEST(ImageBindingPolicy, RejectsObservedShadowGarbageArrayLayerCounts) {
    for (const u32 layers : {3073U, 4097U, 7681U}) {
        EXPECT_FALSE(IsImageBindingExtentUsable(
            ImageBindingKind::TwoDimensional, {128, 128, 1, layers}, TestImageLimits));
    }
}

TEST(ImageBindingPolicy, RejectsMipCountsBeyondTheCompleteChain) {
    EXPECT_TRUE(IsImageBindingExtentUsable(
        ImageBindingKind::TwoDimensional,
        {.width = 1024, .height = 1, .depth = 1, .layers = 1, .levels = 11},
        TestImageLimits));
    EXPECT_FALSE(IsImageBindingExtentUsable(
        ImageBindingKind::TwoDimensional,
        {.width = 1, .height = 1, .depth = 1, .layers = 1, .levels = 10},
        TestImageLimits));
    EXPECT_FALSE(IsImageBindingExtentUsable(
        ImageBindingKind::ThreeDimensional,
        {.width = 8, .height = 8, .depth = 8, .layers = 1, .levels = 5},
        TestImageLimits));
}

TEST(ImageBindingPolicy, RejectsDimensionOverflowAndInvalidShape) {
    EXPECT_FALSE(IsImageBindingExtentUsable(
        ImageBindingKind::OneDimensional, {16385, 1, 1, 1}, TestImageLimits));
    EXPECT_FALSE(IsImageBindingExtentUsable(
        ImageBindingKind::TwoDimensional, {128, 16385, 1, 1}, TestImageLimits));
    EXPECT_FALSE(IsImageBindingExtentUsable(
        ImageBindingKind::ThreeDimensional, {128, 128, 2049, 1}, TestImageLimits));
    EXPECT_FALSE(IsImageBindingExtentUsable(
        ImageBindingKind::Cube, {256, 128, 1, 6}, TestImageLimits));
    EXPECT_FALSE(IsImageBindingExtentUsable(
        ImageBindingKind::Cube, {256, 256, 1, 2049}, TestImageLimits));
    EXPECT_FALSE(IsImageBindingExtentUsable(
        ImageBindingKind::TwoDimensional, {0, 128, 1, 1}, TestImageLimits));
}

TEST(ImageBindingPolicy, UsesTheRuntimeMappingDomainForGuestAdmission) {
    constexpr VAddr address = 0x12345000;
    constexpr u64 required_size = 0x4000;
    const ImageBindingExtent extent{.width = 64, .height = 64, .depth = 1, .layers = 1};

    EXPECT_TRUE(IsGuestImageBindingUsable(address, 63, 63, ImageBindingKind::TwoDimensional,
                                          extent, required_size, TestImageLimits,
                                          [](VAddr candidate, u64 size) {
                                              return candidate == address && size == 1;
                                          }));
    EXPECT_FALSE(IsGuestImageBindingUsable(address, 63, 63, ImageBindingKind::TwoDimensional,
                                           extent, required_size, TestImageLimits,
                                           [](VAddr, u64) { return false; }));
    EXPECT_FALSE(IsGuestImageBindingUsable(address, 62, 63, ImageBindingKind::TwoDimensional,
                                           extent, required_size, TestImageLimits,
                                           [](VAddr, u64) { return true; }));
}

TEST(ImageBindingPolicy, RejectsWrappedOrGuestImpossibleLargeLayouts) {
    constexpr VAddr address = 0x240000000;
    constexpr u64 observed_shadow_layout_size = 12165749760ULL;
    const ImageBindingExtent extent{
        .width = 1920, .height = 1080, .depth = 1, .layers = 273, .levels = 1};

    bool mapping_queried = false;
    EXPECT_FALSE(IsGuestImageBindingUsable(
        address, 40959, 1919, ImageBindingKind::TwoDimensional, extent,
        observed_shadow_layout_size, TestImageLimits, [&](VAddr candidate, u64 size) {
            mapping_queried = true;
            return candidate == address && size == 1;
        }));
    EXPECT_FALSE(mapping_queried);

    // Pipeline specialization describes the binding as a whole, while runtime descriptor
    // admission handles each dynamically indexed array entry. A single corrupt runtime entry
    // must not erase an otherwise valid image-array specialization and black out every entry.
    EXPECT_TRUE(IsGuestImageSpecializationUsable(
        address, 40959, 1919, ImageBindingKind::TwoDimensional, extent, TestImageLimits,
        [](VAddr candidate, u64 size) { return candidate == address && size == 1; }));

    EXPECT_FALSE(IsGuestImageBindingUsable(
        std::numeric_limits<VAddr>::max() - 15, 63, 63, ImageBindingKind::TwoDimensional,
        {.width = 64, .height = 64, .depth = 1, .layers = 1}, 32, TestImageLimits,
        [](VAddr, u64) { return true; }));
}

} // namespace
} // namespace VideoCore

namespace Shader {
namespace {

TEST(MsaaImageArgumentPolicy, SelectsSampleAfterCoordinatesWithoutMipModifier) {
    const auto two_dimensional = PlanMsaaImageArgument(2, 4, false, false);
    EXPECT_TRUE(two_dimensional.valid);
    EXPECT_FALSE(two_dimensional.drop_zero_mip);
    EXPECT_FALSE(two_dimensional.force_zero_sample);
    EXPECT_EQ(two_dimensional.sample_argument, 2U);

    const auto array = PlanMsaaImageArgument(3, 4, false, false);
    EXPECT_TRUE(array.valid);
    EXPECT_FALSE(array.drop_zero_mip);
    EXPECT_FALSE(array.force_zero_sample);
    EXPECT_EQ(array.sample_argument, 3U);
}

TEST(MsaaImageArgumentPolicy, NeverReadsPastTheFourTranslatedAddressArguments) {
    const auto two_dimensional = PlanMsaaImageArgument(2, 4, true, true);
    EXPECT_TRUE(two_dimensional.valid);
    EXPECT_TRUE(two_dimensional.drop_zero_mip);
    EXPECT_FALSE(two_dimensional.force_zero_sample);
    EXPECT_EQ(two_dimensional.sample_argument, 3U);

    const auto array = PlanMsaaImageArgument(3, 4, true, true);
    EXPECT_TRUE(array.valid);
    EXPECT_TRUE(array.drop_zero_mip);
    EXPECT_TRUE(array.force_zero_sample);
    EXPECT_EQ(array.sample_argument, 3U);
}

TEST(MsaaImageArgumentPolicy, RejectsNonZeroOrDynamicMipForMultisampledImages) {
    const auto invalid = PlanMsaaImageArgument(3, 4, true, false);
    EXPECT_FALSE(invalid.valid);

    const auto missing_sample = PlanMsaaImageArgument(3, 3, false, false);
    EXPECT_FALSE(missing_sample.valid);
}

} // namespace
} // namespace Shader
