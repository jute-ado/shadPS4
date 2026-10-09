// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <utility>

#include <vulkan/vulkan.hpp>

namespace VideoCore {

struct SupportedImageUsage {
    vk::ImageUsageFlags usage{};
    bool dropped_optional_storage{};
};

[[nodiscard]] constexpr vk::FormatFeatureFlags2 RequiredImageBindingFormatFeatures(
    const bool is_storage) {
    return is_storage ? vk::FormatFeatureFlagBits2::eStorageImage |
                            vk::FormatFeatureFlagBits2::eStorageReadWithoutFormat |
                            vk::FormatFeatureFlagBits2::eStorageWriteWithoutFormat
                      : vk::FormatFeatureFlagBits2::eSampledImage;
}

[[nodiscard]] constexpr bool IsImageBindingFormatUsable(
    const bool is_storage, const vk::FormatFeatureFlags2 format_features) {
    if (!is_storage) {
        return true;
    }
    const auto required = RequiredImageBindingFormatFeatures(true);
    return (format_features & required) == required;
}

template <typename SupportsUsage>
[[nodiscard]] std::optional<SupportedImageUsage> SelectSupportedImageUsage(
    const bool is_block_encoded, const vk::ImageUsageFlags preferred_usage,
    SupportsUsage&& supports_usage) {
    if (std::forward<SupportsUsage>(supports_usage)(preferred_usage)) {
        return SupportedImageUsage{.usage = preferred_usage};
    }

    if (!is_block_encoded || !(preferred_usage & vk::ImageUsageFlagBits::eStorage)) {
        return std::nullopt;
    }

    const auto required_usage = preferred_usage & ~vk::ImageUsageFlagBits::eStorage;
    if (!std::forward<SupportsUsage>(supports_usage)(required_usage)) {
        return std::nullopt;
    }
    return SupportedImageUsage{
        .usage = required_usage,
        .dropped_optional_storage = true,
    };
}

} // namespace VideoCore
