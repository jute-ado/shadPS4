// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <ranges>
#include <vector>

#include "common/types.h"

namespace VideoCore {

struct VertexBufferRange {
    VAddr base_address;
    VAddr end_address;
};

template <std::ranges::input_range Buffers, typename ClampSize>
[[nodiscard]] std::vector<VertexBufferRange> BuildVertexBufferRanges(const Buffers& buffers,
                                                                     ClampSize&& clamp_size) {
    std::vector<VertexBufferRange> ranges;
    for (const auto& buffer : buffers) {
        const u64 requested_size = buffer.GetSize();
        if (requested_size == 0) {
            continue;
        }
        const u64 clamped_size = std::invoke(clamp_size, buffer.base_address, requested_size);
        const u64 address_space_left = std::numeric_limits<VAddr>::max() - buffer.base_address;
        const u64 safe_size = std::min(clamped_size, address_space_left);
        if (safe_size != 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + safe_size);
        }
    }

    std::ranges::sort(ranges, {}, &VertexBufferRange::base_address);
    std::vector<VertexBufferRange> merged;
    for (const auto& range : ranges) {
        if (merged.empty() || merged.back().end_address < range.base_address) {
            merged.emplace_back(range);
        } else {
            merged.back().end_address = std::max(merged.back().end_address, range.end_address);
        }
    }
    return merged;
}

} // namespace VideoCore
