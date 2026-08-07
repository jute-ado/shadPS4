// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_set>

#include "common/types.h"

namespace AmdGpu {

struct OcclusionQueryReuseSnapshot {
    u64 sequence{};
    u64 dumps{};
    u64 fresh_targets{};
    u64 reused_targets{};
    u64 unknown_targets{};
    u64 no_prior_valid{};
    u64 partial_prior_valid{};
    u64 all_prior_valid{};
    u64 distinct_targets{};
    u32 min_counter_pairs{};
    u32 max_counter_pairs{};
};

class OcclusionQueryReuseDiagnostic {
public:
    static constexpr u64 ValidMask = 1ULL << 63;

    OcclusionQueryReuseDiagnostic(u64 report_interval_, u64 report_limit_, u64 target_limit_)
        : report_interval{std::max<u64>(report_interval_, 1)}, report_limit{report_limit_},
          target_limit{target_limit_} {
        seen_targets.reserve(static_cast<size_t>(target_limit));
    }

    [[nodiscard]] std::optional<OcclusionQueryReuseSnapshot> Observe(
        std::uintptr_t target, const u64* results, u32 counter_pairs) {
        if (reports_emitted >= report_limit) {
            return std::nullopt;
        }

        if (seen_targets.contains(target)) {
            ++current.reused_targets;
        } else if (seen_targets.size() < target_limit) {
            seen_targets.insert(target);
            ++current.fresh_targets;
        } else {
            ++current.unknown_targets;
        }

        u32 valid_count{};
        for (u32 i = 0; i < counter_pairs; ++i) {
            valid_count += (results[i * 2] & ValidMask) != 0;
        }
        if (valid_count == 0) {
            ++current.no_prior_valid;
        } else if (valid_count == counter_pairs) {
            ++current.all_prior_valid;
        } else {
            ++current.partial_prior_valid;
        }

        current.min_counter_pairs = std::min(current.min_counter_pairs, counter_pairs);
        current.max_counter_pairs = std::max(current.max_counter_pairs, counter_pairs);
        ++current.dumps;
        if (current.dumps < report_interval) {
            return std::nullopt;
        }

        current.sequence = ++reports_emitted;
        current.distinct_targets = seen_targets.size();
        const auto snapshot = current;
        current = {};
        current.min_counter_pairs = std::numeric_limits<u32>::max();
        return snapshot;
    }

private:
    const u64 report_interval;
    const u64 report_limit;
    const u64 target_limit;
    u64 reports_emitted{};
    std::unordered_set<std::uintptr_t> seen_targets;
    OcclusionQueryReuseSnapshot current{.min_counter_pairs = std::numeric_limits<u32>::max()};
};

} // namespace AmdGpu
