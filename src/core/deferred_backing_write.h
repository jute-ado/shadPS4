// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <shared_mutex>
#include <span>
#include <utility>
#include <vector>

#include "common/types.h"
#include "core/memory_range_identity.h"

namespace Core {

struct ReadbackWriteRange {
    VAddr address;
    u64 size;
    u64 source_offset;
    bool operator==(const ReadbackWriteRange&) const = default;
};

// Owned metadata only: no guest pointers, map iterators, or references to mutable VMA objects.
// A token is an incarnation check, not a lease retaining an unmapped guest allocation.
struct DeferredBackingWrite {
    struct Fragment {
        VAddr address;
        PAddr physical_address;
        u64 size;
        u64 source_offset;
        u64 incarnation;
        bool operator==(const Fragment&) const = default;
    };

    std::vector<Fragment> fragments;
    u64 source_size{};
    bool operator==(const DeferredBackingWrite&) const = default;
};

// The caller holds its mapping lock. Enqueue admits only fully backed destinations. Completion
// may publish fewer bytes if ownership changes, without discarding other still-owned bytes.
template <typename VmaMap>
[[nodiscard]] std::optional<DeferredBackingWrite> CaptureDeferredBackingWrite(
    const VmaMap& areas, const MemoryRangeIdentity& incarnations,
    std::span<const ReadbackWriteRange> ranges, u64 source_size) {
    DeferredBackingWrite token;
    token.source_size = source_size;
    for (const auto& range : ranges) {
        if (range.size > std::numeric_limits<VAddr>::max() - range.address ||
            range.source_offset > source_size || range.size > source_size - range.source_offset) {
            return std::nullopt;
        }
        auto incarnation = incarnations.Capture(range.address, range.size);
        if (!incarnation || std::ranges::any_of(incarnation->parts, [](const auto& part) {
                return part.identity == 0;
            })) {
            return std::nullopt;
        }
        size_t identity_index = 0;
        const VAddr end = range.address + range.size;
        VAddr cursor = range.address;
        auto area = areas.upper_bound(cursor);
        if (area == areas.begin() && cursor != end) {
            return std::nullopt;
        }
        if (area != areas.begin()) {
            --area;
        }
        while (cursor < end) {
            if (area == areas.end()) {
                return std::nullopt;
            }
            const auto& vma = area->second;
            if (area->first != vma.base || !vma.HasPhysicalBacking() || vma.base > cursor ||
                vma.size > std::numeric_limits<VAddr>::max() - vma.base ||
                cursor - vma.base >= vma.size) {
                return std::nullopt;
            }
            auto physical = vma.phys_areas.upper_bound(cursor - vma.base);
            if (physical == vma.phys_areas.begin()) {
                return std::nullopt;
            }
            --physical;
            while (cursor < std::min(end, vma.base + vma.size)) {
                if (physical == vma.phys_areas.end()) {
                    return std::nullopt;
                }
                const auto& [virtual_offset, backing] = *physical;
                const auto offset = cursor - vma.base;
                if (virtual_offset > offset || virtual_offset > vma.size ||
                    backing.size > vma.size - virtual_offset ||
                    backing.size > std::numeric_limits<PAddr>::max() - backing.base ||
                    offset - virtual_offset >= backing.size) {
                    return std::nullopt;
                }
                const auto& identity = incarnation->parts[identity_index];
                const u64 identity_end = identity.address + identity.size;
                const u64 physical_end = vma.base + virtual_offset + backing.size;
                const u64 size =
                    std::min({end - cursor, physical_end - cursor, identity_end - cursor});
                const DeferredBackingWrite::Fragment fragment{
                    cursor, backing.base + (offset - virtual_offset), size,
                    range.source_offset + (cursor - range.address), identity.identity};
                if (!token.fragments.empty() &&
                    token.fragments.back().incarnation == fragment.incarnation &&
                    token.fragments.back().address + token.fragments.back().size ==
                        fragment.address &&
                    token.fragments.back().physical_address + token.fragments.back().size ==
                        fragment.physical_address &&
                    token.fragments.back().source_offset + token.fragments.back().size ==
                        fragment.source_offset) {
                    // VMA/physical-map splitting and coalescing do not change backing ownership.
                    token.fragments.back().size += size;
                } else {
                    token.fragments.push_back(fragment);
                }
                cursor += size;
                if (cursor == physical_end) {
                    ++physical;
                }
                if (cursor == identity_end) {
                    ++identity_index;
                }
            }
            ++area;
        }
    }
    return token;
}

// Preserve each still-owned byte even when another part of the download has been replaced.
// Validate source and host extents before any copy. Check, publication, and on_published share
// the VMM lock; on_published receives only published ranges and must not reenter the VMM or wait
// for guest/GPU work. False means some bytes were refused, not necessarily that nothing copied.
template <typename Mutex, typename VmaMap, typename OnPublished>
[[nodiscard]] bool PublishDeferredBackingWrite(Mutex& mutex, const VmaMap& areas,
                                               const MemoryRangeIdentity& incarnations,
                                               u8* backing_base, const DeferredBackingWrite& token,
                                               std::span<const u8> source,
                                               OnPublished&& on_published) {
    std::shared_lock lock{mutex};
    if (source.size() != token.source_size) {
        return false;
    }
    constexpr auto Max = std::numeric_limits<uintptr_t>::max();
    const auto backing_address = reinterpret_cast<uintptr_t>(backing_base);
    const auto source_address = reinterpret_cast<uintptr_t>(source.data());
    if ((!token.fragments.empty() && (!backing_base || !source.data())) ||
        source.size() > Max - source_address) {
        return false;
    }
    bool complete = true;
    std::vector<DeferredBackingWrite::Fragment> published;
    for (const auto& captured : token.fragments) {
        const auto current_owners = incarnations.Capture(captured.address, captured.size);
        if (!current_owners) {
            return false;
        }
        for (const auto& owner : current_owners->parts) {
            if (owner.identity != captured.incarnation) {
                complete = false;
                continue;
            }
            // A later partial replacement can split one originally contiguous incarnation.
            // Check each surviving interval against its original physical destination.
            const ReadbackWriteRange range{owner.address, owner.size,
                                           captured.source_offset +
                                               (owner.address - captured.address)};
            const auto current = CaptureDeferredBackingWrite(areas, incarnations,
                                                             std::span{&range, 1}, source.size());
            if (!current || !std::ranges::all_of(current->fragments, [&](const auto& fragment) {
                    return fragment.incarnation == captured.incarnation &&
                           fragment.physical_address ==
                               captured.physical_address + (fragment.address - captured.address);
                })) {
                complete = false;
                continue;
            }
            published.insert(published.end(), current->fragments.begin(), current->fragments.end());
        }
    }
    for (const auto& fragment : published) {
        if (fragment.physical_address > Max - backing_address ||
            fragment.size > Max - backing_address - fragment.physical_address) {
            return false;
        }
    }
    std::vector<ReadbackWriteRange> published_ranges;
    for (const auto& fragment : published) {
        if (!published_ranges.empty() &&
            published_ranges.back().address + published_ranges.back().size == fragment.address &&
            published_ranges.back().source_offset + published_ranges.back().size ==
                fragment.source_offset) {
            published_ranges.back().size += fragment.size;
        } else {
            published_ranges.push_back({fragment.address, fragment.size, fragment.source_offset});
        }
    }
    for (const auto& fragment : published) {
        std::memcpy(backing_base + fragment.physical_address,
                    source.data() + fragment.source_offset, fragment.size);
    }
    if (!published_ranges.empty()) {
        std::invoke(std::forward<OnPublished>(on_published),
                    std::span<const ReadbackWriteRange>{published_ranges});
    }
    return complete;
}

} // namespace Core
