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

[[nodiscard]] constexpr bool ShouldAllowDiagnosticIncompleteImageBacking(
    const bool has_lod_feedback, const bool is_written, const bool diagnostic_enabled) {
    return diagnostic_enabled && has_lod_feedback && !is_written;
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
