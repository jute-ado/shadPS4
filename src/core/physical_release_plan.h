// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <vector>

namespace Core {

struct MemoryReleaseSpan {
    std::uint64_t address;
    std::uint64_t size;
    bool operator==(const MemoryReleaseSpan&) const = default;
};

inline std::optional<MemoryReleaseSpan> IntersectMemoryReleaseRanges(
    std::uint64_t address, std::uint64_t size, std::uint64_t other, std::uint64_t other_size) {
    if (size > std::numeric_limits<std::uint64_t>::max() - address ||
        other_size > std::numeric_limits<std::uint64_t>::max() - other) {
        return std::nullopt;
    }
    if (size == 0 || other_size == 0 || address + size <= other || other + other_size <= address) {
        return std::nullopt;
    }
    const auto start = std::max(address, other);
    return MemoryReleaseSpan{start, std::min(address + size, other + other_size) - start};
}

template <typename PhysMap, typename IsFree>
std::optional<std::vector<MemoryReleaseSpan>> PlanPhysicalMemoryRelease(
    const PhysMap& areas, std::uint64_t address, std::uint64_t size, bool checked, IsFree&& is_free) {
    if (size > std::numeric_limits<std::uint64_t>::max() - address) {
        return std::nullopt;
    }
    std::vector<MemoryReleaseSpan> result;
    if (size == 0) {
        return result;
    }
    const auto end = address + size;
    auto cursor = address;
    auto area = areas.upper_bound(address);
    if (area != areas.begin()) --area;
    for (; area != areas.end() && area->first < end; ++area) {
        if (area->second.size > std::numeric_limits<std::uint64_t>::max() - area->first) {
            return std::nullopt;
        }
        const auto span = IntersectMemoryReleaseRanges(address, size, area->first, area->second.size);
        if (!span) {
            continue;
        }
        if (span->address < cursor || (checked && span->address != cursor)) {
            return std::nullopt;
        }
        cursor = span->address + span->size;
        if (is_free(area->second)) {
            if (checked) return std::nullopt;
            continue;
        }
        result.push_back(*span);
    }
    if (checked && cursor != end) {
        return std::nullopt;
    }
    return result;
}

} // namespace Core
