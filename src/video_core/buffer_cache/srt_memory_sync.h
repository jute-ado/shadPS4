// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>
#include "common/alignment.h"
#include "common/types.h"

namespace VideoCore {

template <typename PageManager, typename BufferCache>
bool ReadSrtMemoryRange(PageManager& page_manager, BufferCache& buffer_cache, VAddr address,
                        void* output, u64 size) {
    constexpr VAddr MaxGpuAddress = 1ULL << 40;
    if (address == 0 || output == nullptr || size == 0 ||
        size > std::numeric_limits<u32>::max() ||
        address >= MaxGpuAddress || size > MaxGpuAddress - address) {
        return false;
    }
    constexpr u64 BufferCachePageSize = 16_KB;
    const VAddr mapped_begin = Common::AlignDown(address, BufferCachePageSize);
    const VAddr mapped_end = Common::AlignUp(address + size, BufferCachePageSize);
    if (!page_manager.IsGpuMapped(mapped_begin, mapped_end - mapped_begin)) {
        return false;
    }
    // Snapshot into walker-owned storage. Publishing a GPU readback into the live guest address
    // space can overwrite newer CPU writes while other guest threads continue running.
    return buffer_cache.ReadMemorySnapshot(address, output, size);
}

template <typename PageManager, typename BufferCache>
class SrtMemoryReadSession {
public:
    SrtMemoryReadSession(PageManager& page_manager_, BufferCache& buffer_cache_)
        : page_manager{page_manager_}, buffer_cache{buffer_cache_} {}

    [[nodiscard]] bool Read(VAddr address, void* output, u64 size) {
        return ReadSrtMemoryRange(page_manager, *this, address, output, size);
    }

    static bool PS4_SYSV_ABI Callback(void* context, VAddr address, void* output, u64 size) {
        return static_cast<SrtMemoryReadSession*>(context)->Read(address, output, size);
    }

    [[nodiscard]] bool ReadMemorySnapshot(VAddr address, void* output, u64 size) {
        constexpr u64 TrackingPageSize = 16_KB;
        const VAddr page_address = Common::AlignDown(address, TrackingPageSize);
        const bool fits_one_page = address + size <= page_address + TrackingPageSize;
        if (!fits_one_page) {
            return buffer_cache.ReadMemorySnapshot(address, output, size);
        }

        auto page = pages.find(page_address);
        if (page == pages.end()) {
            std::vector<u8> bytes(TrackingPageSize);
            if (!buffer_cache.ReadMemorySnapshot(page_address, bytes.data(), bytes.size())) {
                return false;
            }
            page = pages.emplace(page_address, std::move(bytes)).first;
        }
        const u64 page_offset = address - page_address;
        std::memcpy(output, page->second.data() + page_offset, size);
        return true;
    }

private:
    PageManager& page_manager;
    BufferCache& buffer_cache;
    std::unordered_map<VAddr, std::vector<u8>> pages;
};

} // namespace VideoCore
