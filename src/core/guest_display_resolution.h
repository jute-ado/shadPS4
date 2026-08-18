// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace Core {

struct GuestDisplayResolution {
    std::uint32_t width;
    std::uint32_t height;

    constexpr bool operator==(const GuestDisplayResolution&) const = default;
};

inline constexpr std::array GuestDisplayResolutionPresets{
    GuestDisplayResolution{1280, 720},
    GuestDisplayResolution{1920, 1080},
    GuestDisplayResolution{2560, 1440},
    GuestDisplayResolution{3840, 2160},
};
inline constexpr std::size_t CustomGuestDisplayResolutionPreset =
    GuestDisplayResolutionPresets.size();

[[nodiscard]] constexpr std::size_t FindGuestDisplayResolutionPreset(const std::uint32_t width,
                                                                      const std::uint32_t height) {
    for (std::size_t index = 0; index < GuestDisplayResolutionPresets.size(); ++index) {
        if (GuestDisplayResolutionPresets[index] == GuestDisplayResolution{width, height}) {
            return index;
        }
    }
    return CustomGuestDisplayResolutionPreset;
}

[[nodiscard]] constexpr GuestDisplayResolution ResolveGuestDisplayResolutionPreset(
    const std::size_t preset, const GuestDisplayResolution custom) {
    return preset < GuestDisplayResolutionPresets.size() ? GuestDisplayResolutionPresets[preset]
                                                         : custom;
}

[[nodiscard]] inline std::string FormatGuestDisplayResolution(
    const GuestDisplayResolution resolution) {
    return "Guest display resolution: " + std::to_string(resolution.width) + " x " +
           std::to_string(resolution.height);
}

} // namespace Core
