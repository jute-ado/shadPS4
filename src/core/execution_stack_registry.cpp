// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/execution_stack_registry.h"

#include <algorithm>
#include <limits>
#include <mutex>

namespace Core {
namespace {

constexpr VAddr PageSize = 4_KB;
constexpr VAddr PageMask = PageSize - 1;

[[nodiscard]] bool IsValidRange(VAddr address, u64 size) {
    return size != 0 && address <= std::numeric_limits<VAddr>::max() - size;
}

[[nodiscard]] VAddr RangeEnd(const ExecutionStackRange& range) {
    return range.address + range.size;
}

} // namespace

bool ExecutionStackRegistry::Register(VAddr address, u64 size) {
    if (!IsValidRange(address, size)) {
        return false;
    }

    const VAddr end = address + size;
    if (end > std::numeric_limits<VAddr>::max() - PageMask) {
        return false;
    }

    std::unique_lock lock{mutex};
    ++registrations[{address, size}];
    RebuildMergedRanges();
    return true;
}

bool ExecutionStackRegistry::Unregister(VAddr address, u64 size) {
    if (!IsValidRange(address, size)) {
        return false;
    }

    std::unique_lock lock{mutex};
    const auto registration = registrations.find({address, size});
    if (registration == registrations.end()) {
        return false;
    }

    if (--registration->second == 0) {
        registrations.erase(registration);
    }
    RebuildMergedRanges();
    return true;
}

bool ExecutionStackRegistry::Overlaps(VAddr address, u64 size) const {
    if (!IsValidRange(address, size)) {
        return false;
    }

    const VAddr end = address + size;
    std::shared_lock lock{mutex};
    const auto first = std::lower_bound(
        merged_ranges.begin(), merged_ranges.end(), address,
        [](const ExecutionStackRange& range, VAddr value) { return RangeEnd(range) <= value; });
    return first != merged_ranges.end() && first->address < end;
}

std::vector<ExecutionStackRange> ExecutionStackRegistry::GetExcludedRanges(VAddr address,
                                                                            u64 size) const {
    if (!IsValidRange(address, size)) {
        return {};
    }

    const VAddr end = address + size;
    std::vector<ExecutionStackRange> result;
    std::shared_lock lock{mutex};
    auto stack = std::lower_bound(
        merged_ranges.begin(), merged_ranges.end(), address,
        [](const ExecutionStackRange& range, VAddr value) { return RangeEnd(range) <= value; });
    for (; stack != merged_ranges.end() && stack->address < end; ++stack) {
        const VAddr intersection_begin = std::max(address, stack->address);
        const VAddr intersection_end = std::min(end, RangeEnd(*stack));
        result.push_back({intersection_begin, intersection_end - intersection_begin});
    }
    return result;
}

std::vector<ExecutionStackRange> ExecutionStackRegistry::GetWatchableRanges(VAddr address,
                                                                             u64 size) const {
    if (!IsValidRange(address, size)) {
        return {};
    }

    const VAddr end = address + size;
    VAddr cursor = address;
    std::vector<ExecutionStackRange> result;
    std::shared_lock lock{mutex};
    auto stack = std::lower_bound(
        merged_ranges.begin(), merged_ranges.end(), address,
        [](const ExecutionStackRange& range, VAddr value) { return RangeEnd(range) <= value; });
    for (; stack != merged_ranges.end() && stack->address < end; ++stack) {
        if (cursor < stack->address) {
            result.push_back({cursor, std::min(end, stack->address) - cursor});
        }
        cursor = std::max(cursor, RangeEnd(*stack));
        if (cursor >= end) {
            break;
        }
    }
    if (cursor < end) {
        result.push_back({cursor, end - cursor});
    }
    return result;
}

void ExecutionStackRegistry::RebuildMergedRanges() {
    merged_ranges.clear();
    for (const auto& [registration, references] : registrations) {
        if (references == 0) {
            continue;
        }

        const auto [address, size] = registration;
        const VAddr aligned_address = address & ~PageMask;
        const VAddr aligned_end = (address + size + PageMask) & ~PageMask;
        if (merged_ranges.empty() || RangeEnd(merged_ranges.back()) < aligned_address) {
            merged_ranges.push_back({aligned_address, aligned_end - aligned_address});
            continue;
        }

        auto& previous = merged_ranges.back();
        previous.size = std::max(RangeEnd(previous), aligned_end) - previous.address;
    }
}

ExecutionStackRegistry& GetExecutionStackRegistry() {
    static ExecutionStackRegistry registry;
    return registry;
}

} // namespace Core
