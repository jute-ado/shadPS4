// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace Vulkan {

struct CapturedSrtRootProvenance {
    std::array<std::uint32_t, 16> words{};
    std::size_t declared_ud_dwords{};
    std::size_t flat_dwords{};
    std::uint32_t copied_dwords{};
    bool has_full_prefix{};

    bool operator==(const CapturedSrtRootProvenance&) const = default;
};

// Only the owned flattened prefix is copied. Declared count is metadata, not permission
// to read through a borrowed user-data span. A full prefix does not prove mapping ownership.
[[nodiscard]] inline CapturedSrtRootProvenance CaptureSrtRootProvenance(
    std::span<const std::uint32_t> flattened, std::size_t declared_ud_dwords) noexcept {
    CapturedSrtRootProvenance result{};
    result.declared_ud_dwords = declared_ud_dwords;
    result.flat_dwords = flattened.size();
    result.copied_dwords = static_cast<std::uint32_t>(std::min(flattened.size(), result.words.size()));
    result.has_full_prefix = result.copied_dwords == result.words.size();
    for (std::size_t i = 0; i < result.copied_dwords; ++i) {
        result.words[i] = flattened[i];
    }
    return result;
}

} // namespace Vulkan
