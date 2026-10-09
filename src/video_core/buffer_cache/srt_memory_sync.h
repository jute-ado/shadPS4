// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include "common/alignment.h"
#include "common/types.h"

namespace VideoCore {

template <typename PageManager, typename BufferCache>
bool SynchronizeSrtMemoryRange(PageManager& page_manager, BufferCache& buffer_cache,
                               VAddr address, u64 size) {
    constexpr VAddr MaxGpuAddress = 1ULL << 40;
    if (address == 0 || size == 0 || size > std::numeric_limits<u32>::max() ||
        address >= MaxGpuAddress || size > MaxGpuAddress - address) {
        return false;
    }
    constexpr u64 BufferCachePageSize = 16_KB;
    const VAddr mapped_begin = Common::AlignDown(address, BufferCachePageSize);
    const VAddr mapped_end = Common::AlignUp(address + size, BufferCachePageSize);
    if (!page_manager.IsGpuMapped(mapped_begin, mapped_end - mapped_begin)) {
        return false;
    }
    // The page tracker is intentionally coarse. SRT fields often share a tracker page with an
    // unrelated GPU-written range, so use the cache's byte-precise ownership set before forcing
    // a synchronous GPU readback.
    if (!buffer_cache.IsRegionGpuModifiedExact(address, size)) {
        return true;
    }
    // SRT walkers commonly request several adjacent 4- or 8-byte fields. Synchronize the
    // containing tracker window so the first dirty field refreshes its neighbours and subsequent
    // walker callbacks do not submit a GPU readback for every individual descriptor word.
    return buffer_cache.ReadMemoryExact(mapped_begin, mapped_end - mapped_begin);
}

} // namespace VideoCore
