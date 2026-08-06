// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

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
bool MakeDmaFaultRangeResident(BufferCache& buffer_cache, VAddr start, u32 size) {
    if (!buffer_cache.TransitionAuthoritativeTextureForDmaRead(start, size)) {
        return false;
    }
    buffer_cache.FindBuffer(start, size);
    buffer_cache.SynchronizeBuffersInRange(start, size);
    return true;
}

} // namespace VideoCore
