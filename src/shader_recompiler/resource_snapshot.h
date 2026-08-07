// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <compare>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <ranges>
#include <vector>

#include "common/types.h"

namespace Shader {

struct ResourceSnapshotRange {
    VAddr address;
    u64 size;

    auto operator<=>(const ResourceSnapshotRange&) const = default;
};

static constexpr size_t MaxResourceSnapshotRanges = 256;

struct ResourceGenerationMismatchObservation {
    bool mismatched{};
    bool should_report{};
    u64 occurrence{};
};

class ResourceGenerationMismatchCounter {
public:
    explicit ResourceGenerationMismatchCounter(u64 report_limit_) : report_limit{report_limit_} {}

    template <std::ranges::input_range First, std::ranges::input_range Second>
    ResourceGenerationMismatchObservation Observe(const First& first, const Second& second) {
        if (std::ranges::equal(first, second)) {
            return {};
        }

        const u64 occurrence = total.fetch_add(1, std::memory_order_relaxed) + 1;
        return {
            .mismatched = true,
            .should_report = occurrence <= report_limit,
            .occurrence = occurrence,
        };
    }

    [[nodiscard]] u64 Total() const noexcept {
        return total.load(std::memory_order_relaxed);
    }

private:
    u64 report_limit;
    std::atomic<u64> total{};
};

/**
 * Captures one immutable resource generation while the caller owns every source range.
 *
 * Ranges are checked, sorted, and coalesced before acquisition. Acquisition is all-or-nothing;
 * any partially acquired prefix is released in reverse order and the capture is not invoked.
 * The release callback must not throw.
 */
template <typename Snapshot, std::ranges::input_range RangeList, typename Acquire, typename Capture,
          typename Release>
std::optional<Snapshot> CaptureOwnedResourceSnapshot(RangeList&& source_ranges,
                                                     Acquire&& acquire, Capture&& capture,
                                                     Release&& release) {
    std::vector<ResourceSnapshotRange> ranges;
    if constexpr (std::ranges::sized_range<RangeList>) {
        if (std::ranges::size(source_ranges) > MaxResourceSnapshotRanges) {
            return std::nullopt;
        }
        ranges.reserve(std::ranges::size(source_ranges));
    }

    for (const ResourceSnapshotRange range : source_ranges) {
        if (range.size == 0) {
            continue;
        }
        if (range.address > std::numeric_limits<VAddr>::max() - range.size) {
            return std::nullopt;
        }
        ranges.push_back(range);
        if (ranges.size() > MaxResourceSnapshotRanges) {
            return std::nullopt;
        }
    }

    std::ranges::sort(ranges, {}, &ResourceSnapshotRange::address);

    std::vector<ResourceSnapshotRange> normalized;
    normalized.reserve(ranges.size());
    for (const ResourceSnapshotRange range : ranges) {
        if (normalized.empty()) {
            normalized.push_back(range);
            continue;
        }

        auto& previous = normalized.back();
        const VAddr previous_end = previous.address + previous.size;
        const VAddr range_end = range.address + range.size;
        if (range.address <= previous_end) {
            previous.size = std::max(previous_end, range_end) - previous.address;
        } else {
            normalized.push_back(range);
        }
    }

    size_t acquired_count = 0;
    const auto release_acquired = [&] {
        while (acquired_count != 0) {
            std::invoke(release, normalized[--acquired_count]);
        }
    };

    try {
        for (const ResourceSnapshotRange range : normalized) {
            if (!std::invoke(acquire, range)) {
                release_acquired();
                return std::nullopt;
            }
            ++acquired_count;
        }

        Snapshot snapshot{};
        std::invoke(capture, snapshot);
        release_acquired();
        return snapshot;
    } catch (...) {
        release_acquired();
        throw;
    }
}

} // namespace Shader
