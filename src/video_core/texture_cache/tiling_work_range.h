// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <ranges>

namespace VideoCore {

struct BasicTilingMipRange {
    std::uint64_t offset{};
    std::uint64_t size{};
};

struct TilingWorkRange {
    std::uint32_t num_mips{};
    std::uint64_t buffer_span{};
    std::uint64_t dispatch_size{};
    std::array<std::uint64_t, 16> packed_offsets{};
};

template <std::ranges::input_range MipRanges>
[[nodiscard]] TilingWorkRange ComputeTilingWorkRange(const MipRanges& mips,
                                                     std::uint64_t buffer_capacity) {
    TilingWorkRange result{};
    for (const auto& mip : mips) {
        if (result.num_mips == result.packed_offsets.size() || mip.size == 0 ||
            mip.offset > buffer_capacity || mip.size > buffer_capacity - mip.offset ||
            mip.size > std::numeric_limits<std::uint64_t>::max() - result.dispatch_size) {
            break;
        }
        result.packed_offsets[result.num_mips] = result.dispatch_size;
        ++result.num_mips;
        result.buffer_span = std::max(result.buffer_span, mip.offset + mip.size);
        result.dispatch_size += mip.size;
    }
    return result;
}

} // namespace VideoCore
