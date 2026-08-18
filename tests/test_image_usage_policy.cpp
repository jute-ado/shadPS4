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

} // namespace
} // namespace VideoCore
