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
    if (!buffer_cache.IsRegionGpuModified(address, size)) {
        return true;
    }
    return buffer_cache.ReadMemoryExact(address, size);
}

} // namespace VideoCore
