// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <limits>

#include "common/types.h"

namespace VideoCore {

constexpr bool IsCacheableFaultRange(VAddr start, VAddr end, VAddr address_space_size) {
    return start != 0 && start < address_space_size && end > start && end <= address_space_size &&
           end - start <= std::numeric_limits<u32>::max();
}

template <typename IsMapped>
constexpr bool IsProcessableDmaFaultRange(VAddr start, VAddr end, VAddr address_space_size,
                                          IsMapped&& is_mapped) {
    if (!IsCacheableFaultRange(start, end, address_space_size)) {
        return false;
    }
    return is_mapped(start, end - start);
}

template <typename BufferCache>
void MakeDmaFaultRangeResident(BufferCache& buffer_cache, VAddr start, u32 size,
                               u64 cache_page_size = 16_KB) {
    if (cache_page_size == 0 || !std::has_single_bit(cache_page_size)) {
        return;
    }
    VAddr segment_start = start;
    u32 remaining = size;
    while (remaining != 0) {
        const u64 page_remaining = cache_page_size - (segment_start & (cache_page_size - 1));
        const u32 segment_size = static_cast<u32>(std::min<u64>(remaining, page_remaining));
        buffer_cache.FindBuffer(segment_start, segment_size);
        segment_start += segment_size;
        remaining -= segment_size;
    }
    buffer_cache.SynchronizeBuffersInRange(start, size);
}

} // namespace VideoCore
