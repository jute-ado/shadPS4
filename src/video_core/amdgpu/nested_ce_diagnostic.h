// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "common/types.h"

namespace AmdGpu {

struct NestedCeDiagnosticObservation {
    bool hazard{};
    bool should_report{};
    bool parent_ce_unfinished{};
    u32 ce_count{};
    u32 de_count{};
    s64 difference{};
    u64 occurrence{};
};

class NestedCeDiagnosticTracker {
public:
    explicit NestedCeDiagnosticTracker(u64 report_limit_) : report_limit{report_limit_} {}

    void ObserveTopLevel(bool has_ccb) noexcept {
        top_level_entries.fetch_add(1, std::memory_order_relaxed);
        if (has_ccb) {
            top_level_entries_with_ccb.fetch_add(1, std::memory_order_relaxed);
        }
    }

    NestedCeDiagnosticObservation ObserveNested(u32 ce_count, u32 de_count,
                                                bool parent_ce_unfinished) noexcept {
        const u64 occurrence = nested_entries.fetch_add(1, std::memory_order_relaxed) + 1;
        const bool hazard = ce_count != 0 || de_count != 0 || parent_ce_unfinished;
        if (hazard) {
            nested_hazards.fetch_add(1, std::memory_order_relaxed);
        }
        return {
            .hazard = hazard,
            .should_report = occurrence <= report_limit,
            .parent_ce_unfinished = parent_ce_unfinished,
            .ce_count = ce_count,
            .de_count = de_count,
            .difference = static_cast<s64>(ce_count) - static_cast<s64>(de_count),
            .occurrence = occurrence,
        };
    }

    [[nodiscard]] u64 TopLevelEntries() const noexcept {
        return top_level_entries.load(std::memory_order_relaxed);
    }

    [[nodiscard]] u64 TopLevelEntriesWithCcb() const noexcept {
        return top_level_entries_with_ccb.load(std::memory_order_relaxed);
    }

    [[nodiscard]] u64 NestedEntries() const noexcept {
        return nested_entries.load(std::memory_order_relaxed);
    }

    [[nodiscard]] u64 NestedHazards() const noexcept {
        return nested_hazards.load(std::memory_order_relaxed);
    }

private:
    u64 report_limit;
    std::atomic<u64> top_level_entries{};
    std::atomic<u64> top_level_entries_with_ccb{};
    std::atomic<u64> nested_entries{};
    std::atomic<u64> nested_hazards{};
};

} // namespace AmdGpu
