// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>

namespace VideoCore {

constexpr float HostToGuestFragmentCoordinate(const float host_coordinate,
                                              const std::uint32_t host_scale) noexcept {
    return host_scale > 1 ? host_coordinate / static_cast<float>(host_scale) : host_coordinate;
}

enum class HostImageScaleDisposition : std::uint8_t {
    Identity,
    Scaled,
    UnsupportedScale,
    UnsupportedImage,
    Overflow,
};

struct HostImageScaleDescriptor {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t display_width{};
    std::uint32_t display_height{};
    std::uint32_t scale_percent{100};
    std::uint32_t levels{1};
    std::uint32_t layers{1};
    std::uint32_t samples{1};
    bool is_2d{true};
    bool is_block_compressed{};
    bool supports_blit{true};
    bool is_render_target{};
    bool is_depth_target{};
};

struct HostImageScalePlan {
    std::uint32_t logical_width{};
    std::uint32_t logical_height{};
    std::uint32_t host_width{};
    std::uint32_t host_height{};
    std::uint32_t numerator{1};
    std::uint32_t denominator{1};
    HostImageScaleDisposition disposition{HostImageScaleDisposition::Identity};

    [[nodiscard]] constexpr bool IsScaled() const {
        return disposition == HostImageScaleDisposition::Scaled;
    }
};

[[nodiscard]] constexpr HostImageScalePlan PlanHostImageScale(
    const HostImageScaleDescriptor& desc) noexcept {
    HostImageScalePlan plan{
        .logical_width = desc.width,
        .logical_height = desc.height,
        .host_width = desc.width,
        .host_height = desc.height,
    };
    if (desc.width == 0 || desc.height == 0) {
        plan.disposition = HostImageScaleDisposition::UnsupportedImage;
        return plan;
    }
    if (desc.scale_percent == 100) {
        return plan;
    }
    if (desc.scale_percent != 200) {
        plan.disposition = HostImageScaleDisposition::UnsupportedScale;
        return plan;
    }
    if (!desc.is_2d || desc.is_block_compressed || !desc.supports_blit || desc.levels != 1 ||
        desc.layers == 0 || desc.samples != 1 ||
        (!desc.is_render_target && !desc.is_depth_target)) {
        plan.disposition = HostImageScaleDisposition::UnsupportedImage;
        return plan;
    }
    if (desc.width > std::numeric_limits<std::uint32_t>::max() / 2 ||
        desc.height > std::numeric_limits<std::uint32_t>::max() / 2) {
        plan.disposition = HostImageScaleDisposition::Overflow;
        return plan;
    }
    if (desc.display_width == 0 || desc.display_height == 0 ||
        static_cast<std::uint64_t>(desc.width) * desc.display_height !=
            static_cast<std::uint64_t>(desc.height) * desc.display_width) {
        plan.disposition = HostImageScaleDisposition::UnsupportedImage;
        return plan;
    }
    plan.host_width = desc.width * 2;
    plan.host_height = desc.height * 2;
    plan.numerator = 2;
    plan.disposition = HostImageScaleDisposition::Scaled;
    return plan;
}

[[nodiscard]] constexpr std::uint32_t ScaleHostCoordinate(const std::uint32_t value,
                                                          const HostImageScalePlan& plan) {
    return plan.IsScaled() ? value * plan.numerator : value;
}

[[nodiscard]] constexpr float ScaleHostCoordinate(const float value,
                                                  const HostImageScalePlan& plan) {
    return plan.IsScaled() ? value * static_cast<float>(plan.numerator) : value;
}

[[nodiscard]] constexpr bool ShouldForceNativeHostImageAccess(
    const bool is_storage, const bool uses_integer_coordinates,
    const bool queries_dimensions) noexcept {
    return is_storage || uses_integer_coordinates || queries_dimensions;
}

[[nodiscard]] constexpr std::uint32_t ResolveHostAttachmentScale(
    const std::uint32_t scale_percent, const std::span<const bool> eligible) noexcept {
    return scale_percent == 200 && !eligible.empty() &&
                   std::ranges::all_of(eligible, std::identity{})
               ? 2u
               : 1u;
}

} // namespace VideoCore
