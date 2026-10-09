// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>

#include "common/types.h"

namespace VideoCore {

constexpr u32 StoredDmaFaultAddressCapacity(u64 area_size_bytes) {
    if (area_size_bytes < 2U * sizeof(u64)) {
        return 0;
    }
    const u64 capacity = area_size_bytes / sizeof(u64) - 1U;
    return capacity > std::numeric_limits<u32>::max() ? std::numeric_limits<u32>::max()
                                                      : static_cast<u32>(capacity);
}

constexpr u32 BoundDmaFaultCount(u64 reported_count, u64 area_size_bytes) {
    const u32 capacity = StoredDmaFaultAddressCapacity(area_size_bytes);
    return reported_count > capacity ? capacity : static_cast<u32>(reported_count);
}

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
void MakeDmaFaultRangeResident(BufferCache& buffer_cache, VAddr start, u32 size) {
    buffer_cache.FindBuffer(start, size);
    buffer_cache.SynchronizeBuffersInRange(start, size);
}

} // namespace VideoCore
