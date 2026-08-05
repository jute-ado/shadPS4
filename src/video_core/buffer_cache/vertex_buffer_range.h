// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <functional>
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
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
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

    for (auto& range : merged) {
        const u64 size =
            std::invoke(clamp_size, range.base_address, range.end_address - range.base_address);
        range.end_address = range.base_address + size;
    }
    return merged;
}

} // namespace VideoCore
