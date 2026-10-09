// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace Vulkan {

struct FragmentInputModeSummary {
    std::uint32_t declared_count{};
    std::uint32_t captured_count{};
    std::uint32_t default_mask{};
    std::uint32_t flat_mask{};
};

// Report raw mode bits independently: IsDefault() deliberately hides the default+flat combination.
// This helper neither admits input metadata nor changes specialization/interpolation behavior.
template <typename Input, std::size_t Extent>
[[nodiscard]] constexpr FragmentInputModeSummary SummarizeFragmentInputModes(
    std::uint32_t declared_count, std::span<Input, Extent> inputs) noexcept {
    FragmentInputModeSummary row{.declared_count = declared_count};
    row.captured_count = static_cast<std::uint32_t>(
        std::min({std::size_t{declared_count}, inputs.size(), std::size_t{32}}));
    for (std::uint32_t index = 0; index < row.captured_count; ++index) {
        const auto bit = std::uint32_t{1} << index;
        if (inputs[index].is_default) {
            row.default_mask |= bit;
        }
        if (inputs[index].is_flat) {
            row.flat_mask |= bit;
        }
    }
    return row;
}

} // namespace Vulkan
