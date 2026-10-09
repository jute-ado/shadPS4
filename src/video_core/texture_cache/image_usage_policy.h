// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <utility>

#include "common/types.h"
#include <vulkan/vulkan.hpp>

namespace VideoCore {

struct SupportedImageUsage {
    vk::ImageUsageFlags usage{};
    bool dropped_optional_storage{};
};

/// Select an image usage that is legal for the requested sample count.  Storage is optional for
/// read-only images because the cache normally advertises a broad usage set for future views, but
/// Vulkan disallows multisampled storage images when shaderStorageImageMultisample is unavailable.
/// A descriptor that actually writes the image must fail closed instead of silently losing storage.
[[nodiscard]] constexpr std::optional<SupportedImageUsage> SelectImageUsageForSamples(
    const vk::ImageUsageFlags preferred_usage, const u32 num_samples,
    const bool storage_required, const bool supports_storage_image_multisample) noexcept {
    constexpr auto storage = vk::ImageUsageFlagBits::eStorage;
    if (!(preferred_usage & storage)) {
        return storage_required ? std::nullopt
                                : std::optional<SupportedImageUsage>{{.usage = preferred_usage}};
    }
    if (num_samples <= 1 || supports_storage_image_multisample) {
        return SupportedImageUsage{.usage = preferred_usage};
    }
    if (storage_required) {
        return std::nullopt;
    }
    return SupportedImageUsage{
        .usage = preferred_usage & ~storage,
        .dropped_optional_storage = true,
    };
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
