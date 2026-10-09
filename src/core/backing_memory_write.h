// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstring>

#include "core/memory.h"

namespace Core {

/// Copies the contiguous physically backed prefix and returns true only for a complete write.
/// A false result may have published a valid prefix; existing readback callers rely on this.
/// The caller must keep the VMA map and backing allocation stable throughout this call and supply
/// a valid, non-overlapping source. Physical metadata describes offsets into that backing
/// allocation.
[[nodiscard]] inline bool TryWriteBackingMemory(const std::map<VAddr, VirtualMemoryArea>& areas,
                                                u8* backing_base, VAddr address, const void* data,
                                                u64 size) {
    if (size == 0) {
        return true;
    }
    const auto source_address = reinterpret_cast<uintptr_t>(data);
    const auto backing_address = reinterpret_cast<uintptr_t>(backing_base);
    constexpr auto MaxHostAddress = std::numeric_limits<uintptr_t>::max();
    if (!data || !backing_base || size > std::numeric_limits<VAddr>::max() - address ||
        size > MaxHostAddress - source_address) {
        return false;
    }

    const auto* source = static_cast<const u8*>(data);
    const VAddr range_end = address + size;
    VAddr cursor = address;
    auto area = areas.upper_bound(cursor);
    if (area == areas.begin()) {
        return false;
    }
    --area;
    while (cursor < range_end) {
        if (area == areas.end()) {
            return false;
        }
        const auto& vma = area->second;
        if (area->first != vma.base || !vma.HasPhysicalBacking() || vma.base > cursor ||
            vma.size > std::numeric_limits<VAddr>::max() - vma.base ||
            cursor - vma.base >= vma.size) {
            return false;
        }
        const VAddr vma_end = vma.base + vma.size;
        auto physical = vma.phys_areas.upper_bound(cursor - vma.base);
        if (physical == vma.phys_areas.begin()) {
            return false;
        }
        --physical;
        while (cursor < std::min(range_end, vma_end)) {
            if (physical == vma.phys_areas.end()) {
                return false;
            }
            const auto& [virtual_offset, region] = *physical;
            const u64 offset_in_vma = cursor - vma.base;
            if (virtual_offset > offset_in_vma || virtual_offset > vma.size ||
                region.size > vma.size - virtual_offset ||
                region.size > std::numeric_limits<PAddr>::max() - region.base ||
                offset_in_vma - virtual_offset >= region.size) {
                return false;
            }
            const u64 offset_in_region = offset_in_vma - virtual_offset;
            const u64 copy_size = std::min(range_end - cursor, region.size - offset_in_region);
            const u64 backing_offset = region.base + offset_in_region;
            if (backing_offset > MaxHostAddress - backing_address ||
                copy_size > MaxHostAddress - backing_address - backing_offset) {
                return false;
            }
            std::memcpy(backing_base + backing_offset, source + (cursor - address), copy_size);
            cursor += copy_size;
            ++physical;
        }
        ++area;
    }

    return true;
}

} // namespace Core
