// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace Core::Loader {

template <std::size_t Size>
constexpr void WriteModuleFilename(std::array<char, Size>& destination,
                                   std::string_view module_name) {
    destination.fill('\0');
    if constexpr (Size <= 1) {
        return;
    }

    constexpr std::string_view suffix = ".sprx";
    const std::size_t content_capacity = Size - 1;
    const std::size_t suffix_size = std::min(suffix.size(), content_capacity);
    const std::size_t stem_size =
        std::min(module_name.size(), content_capacity - suffix_size);
    std::copy_n(module_name.begin(), stem_size, destination.begin());
    std::copy_n(suffix.begin(), suffix_size, destination.begin() + stem_size);
}

inline constexpr std::size_t DynamicFingerprintSize = 0x18;

template <std::size_t Size>
constexpr bool CopyModuleFingerprint(std::array<std::uint8_t, Size>& destination,
                                     std::span<const std::uint8_t> dynamic_data,
                                     std::uint64_t offset) {
    destination.fill(0);
    if (offset > dynamic_data.size() ||
        DynamicFingerprintSize > dynamic_data.size() - offset) {
        return false;
    }
    std::copy_n(dynamic_data.begin() + static_cast<std::size_t>(offset),
                std::min(Size, DynamicFingerprintSize), destination.begin());
    return true;
}

} // namespace Core::Loader
