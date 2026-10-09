// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

#include "core/memory_range_identity.h"

namespace Core {

struct MemoryRetirementPlan {
    bool requires_retirement{};
    std::vector<MemoryRangeIdentity::Snapshot> virtual_ranges;
    std::vector<MemoryRangeIdentity::Snapshot> physical_ranges;
};

template <typename VmaMap>
std::optional<MemoryRetirementPlan> CaptureMemoryRetirementPlan(
    const VmaMap& vmas, const MemoryRangeIdentity& virtual_identity,
    const MemoryRangeIdentity& physical_identity, std::uint64_t address, std::uint64_t size,
    bool physical) {
    if (size > std::numeric_limits<std::uint64_t>::max() - address) {
        return std::nullopt;
    }
    MemoryRetirementPlan plan;
    if (size == 0) {
        return plan;
    }
    const auto append = [](auto& ranges, const MemoryRangeIdentity& identity, auto base, auto len) {
        auto snapshot = identity.Capture(base, len);
        if (!snapshot) {
            return false;
        }
        ranges.push_back(std::move(*snapshot));
        return true;
    };
    const auto end = address + size;
    if (physical) {
        append(plan.physical_ranges, physical_identity, address, size);
    } else {
        append(plan.virtual_ranges, virtual_identity, address, size);
    }
    for (const auto& [base, area] : vmas) {
        if (!area.IsMapped()) {
            continue;
        }
        if (area.size > std::numeric_limits<std::uint64_t>::max() - base) {
            return std::nullopt;
        }
        if (!physical && (base >= end || base + area.size <= address)) {
            continue;
        }
        if (!physical) {
            const auto start = std::max(base, address);
            const auto stop = std::min(base + area.size, end);
            // Host GPU registration does not depend on guest GPU protection flags.
            plan.requires_retirement |= start < 0x10000000000ULL;
            for (const auto& [offset, backing] : area.phys_areas) {
                if (offset > area.size || backing.size > area.size - offset ||
                    backing.size > std::numeric_limits<std::uint64_t>::max() - backing.base) {
                    return std::nullopt;
                }
                const auto overlap_start = std::max(start, base + offset);
                const auto overlap_end = std::min(stop, base + offset + backing.size);
                if (overlap_start < overlap_end &&
                    !append(plan.physical_ranges, physical_identity,
                            backing.base + (overlap_start - (base + offset)),
                            overlap_end - overlap_start)) {
                    return std::nullopt;
                }
            }
        } else {
            // Every current alias is copied. Alias insertion/removal stamps physical identity,
            // so revalidation also detects an alias that did not exist during this capture.
            for (const auto& [offset, backing] : area.phys_areas) {
                if (offset > area.size || backing.size > area.size - offset ||
                    backing.size > std::numeric_limits<std::uint64_t>::max() - backing.base) {
                    return std::nullopt;
                }
                const auto start = std::max(address, backing.base);
                const auto stop = std::min(end, backing.base + backing.size);
                if (start < stop) {
                    const auto virtual_start = base + offset + (start - backing.base);
                    if (!append(plan.virtual_ranges, virtual_identity, virtual_start, stop - start)) {
                        return std::nullopt;
                    }
                    plan.requires_retirement |= virtual_start < 0x10000000000ULL;
                }
            }
        }
    }
    return plan;
}

} // namespace Core
