// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <mutex>

#include "video_core/buffer_cache/memory_invalidation_reason.h"
#include "video_core/buffer_cache/region_manager.h"

namespace VideoCore {

// Returns whether the caller must perform its existing readback, after releasing
// the region lock. Keep the GPU check and CPU state transition atomic.
[[nodiscard]] inline bool InvalidateTrackedRegion(RegionManager& manager, u64 offset, size_t size,
                                                  bool readbacks_enabled,
                                                  MemoryInvalidationReason reason,
                                                  auto&& on_preserve,
                                                  auto&& on_mapping_upload) noexcept {
    std::scoped_lock lock{manager.lock};
    const bool gpu_modified = manager.IsRegionModified<Type::GPU>(offset, size);
    if (gpu_modified && readbacks_enabled) {
        return true;
    }
    // Before-images merge ordinary CPU writes with surviving GPU-owned words.
    // Mapping removal/replacement is not a CPU write and may name an unbacked
    // reservation. It must not sample or retain bytes from that VA incarnation.
    if (gpu_modified && reason == MemoryInvalidationReason::CpuWrite) {
        const VAddr manager_addr = manager.GetCpuAddr();
        const VAddr write_begin = manager_addr + offset;
        const VAddr write_end = write_begin + size;
        const size_t first_page = offset / TRACKER_BYTES_PER_PAGE;
        const size_t page_end =
            (offset + size + TRACKER_BYTES_PER_PAGE - 1) / TRACKER_BYTES_PER_PAGE;
        for (size_t page = first_page; page < page_end; ++page) {
            const u64 page_offset = page * TRACKER_BYTES_PER_PAGE;
            if (!manager.IsRegionModified<Type::GPU>(page_offset, TRACKER_BYTES_PER_PAGE)) {
                continue;
            }
            const VAddr page_addr = manager_addr + page_offset;
            const VAddr page_write_begin = std::max(write_begin, page_addr);
            const VAddr page_write_end = std::min(write_end, page_addr + TRACKER_BYTES_PER_PAGE);
            if (!on_preserve(page_addr, page_write_begin - page_addr,
                             page_write_end - page_write_begin)) {
                return true;
            }
        }
    }
    if (reason == MemoryInvalidationReason::MappingChange) {
        on_mapping_upload(manager.GetCpuAddr() + offset, size);
    }
    manager.ChangeRegionState<Type::CPU, true>(manager.GetCpuAddr() + offset, size);
    return false;
}

} // namespace VideoCore
