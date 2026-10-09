// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <vector>

#include <gtest/gtest.h>

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

TEST(ImageUsagePolicy, SampledMsaaDropsStorageWithoutFeature) {
    // Storage is optional for the cache's broad image usage, but Vulkan forbids a multisampled
    // image with STORAGE usage when shaderStorageImageMultisample is unavailable.  A read-only
    // MSAA descriptor must therefore keep the sampled/transfer uses and drop only storage.
    const auto sampled_only = SelectImageUsageForSamples(PreferredUsage, 4U, false, false);
    ASSERT_TRUE(sampled_only.has_value());
    EXPECT_EQ(sampled_only->usage, RequiredSampledUsage);
    EXPECT_TRUE(sampled_only->dropped_optional_storage);

    const auto feature_supported = SelectImageUsageForSamples(PreferredUsage, 4U, false, true);
    ASSERT_TRUE(feature_supported.has_value());
    EXPECT_EQ(feature_supported->usage, PreferredUsage);
    EXPECT_FALSE(feature_supported->dropped_optional_storage);

    const auto required_storage = SelectImageUsageForSamples(PreferredUsage, 4U, true, false);
    EXPECT_FALSE(required_storage.has_value());

    const auto single_sample_storage = SelectImageUsageForSamples(PreferredUsage, 1U, true, false);
    ASSERT_TRUE(single_sample_storage.has_value());
    EXPECT_EQ(single_sample_storage->usage, PreferredUsage);
    EXPECT_FALSE(single_sample_storage->dropped_optional_storage);
}

} // namespace
} // namespace VideoCore
