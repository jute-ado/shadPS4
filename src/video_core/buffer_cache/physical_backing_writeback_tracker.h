// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>

#include "common/types.h"

namespace VideoCore {

/// Retains the newest submitted GPU timeline tick for each canonical physical page.
class PhysicalBackingWritebackTracker {
public:
    static constexpr u64 PageSize = 16_KB;

    [[nodiscard]] bool Record(std::span<const u64> physical_pages, u64 tick) {
        if (tick == 0 || physical_pages.empty()) {
            return false;
        }
        u64 previous_page = 0;
        bool first_page = true;
        for (const u64 page : physical_pages) {
            if ((page & (PageSize - 1)) != 0 || (!first_page && page <= previous_page)) {
                return false;
            }
            previous_page = page;
            first_page = false;
        }
        for (const u64 page : physical_pages) {
            auto& latest_tick = pending_ticks[page];
            latest_tick = std::max(latest_tick, tick);
        }
        return true;
    }

    [[nodiscard]] std::optional<u64> RequiredTick(
        std::span<const u64> physical_pages) const {
        if (!ValidateQuery(physical_pages) || pending_ticks.empty()) {
            return std::nullopt;
        }

        std::optional<u64> highest;
        for (const u64 page : physical_pages) {
            const auto pending = pending_ticks.find(page);
            if (pending == pending_ticks.end()) {
                continue;
            }
            highest = highest ? std::max(*highest, pending->second) : pending->second;
        }
        return highest;
    }

    [[nodiscard]] std::optional<u64> RequiredTickForAll() const {
        std::optional<u64> highest;
        for (const auto& [page, tick] : pending_ticks) {
            highest = highest ? std::max(*highest, tick) : tick;
        }
        return highest;
    }

    [[nodiscard]] std::vector<u64> CompleteThrough(u64 tick) {
        auto completed_pages = PagesCompletingThrough(tick);
        for (const u64 physical_page : completed_pages) {
            pending_ticks.erase(physical_page);
        }
        return completed_pages;
    }

    [[nodiscard]] std::vector<u64> PagesCompletingThrough(u64 tick) const {
        std::vector<u64> completed_pages;
        completed_pages.reserve(pending_ticks.size());
        for (const auto& [physical_page, pending_tick] : pending_ticks) {
            if (pending_tick <= tick) {
                completed_pages.push_back(physical_page);
            }
        }
        std::ranges::sort(completed_pages);
        return completed_pages;
    }

    [[nodiscard]] bool CompletePages(std::span<const u64> physical_pages, u64 tick) {
        for (const u64 physical_page : physical_pages) {
            const auto pending = pending_ticks.find(physical_page);
            if (pending == pending_ticks.end() || pending->second > tick) {
                return false;
            }
        }
        for (const u64 physical_page : physical_pages) {
            pending_ticks.erase(physical_page);
        }
        return true;
    }

    [[nodiscard]] size_t PendingPageCount() const noexcept {
        return pending_ticks.size();
    }

private:
    [[nodiscard]] static bool ValidateQuery(std::span<const u64> physical_pages) {
        if (physical_pages.empty()) {
            return false;
        }
        u64 previous_page = 0;
        bool first_page = true;
        for (const u64 page : physical_pages) {
            if ((page & (PageSize - 1)) != 0 || (!first_page && page <= previous_page)) {
                return false;
            }
            previous_page = page;
            first_page = false;
        }
        return true;
    }

    std::unordered_map<u64, u64> pending_ticks;
};

template <typename Wait>
[[nodiscard]] std::vector<u64> SynchronizePhysicalBackingWritebacks(
    PhysicalBackingWritebackTracker& tracker, std::span<const u64> physical_pages, Wait&& wait) {
    const auto required_tick = tracker.RequiredTick(physical_pages);
    if (!required_tick) {
        return {};
    }
    std::invoke(std::forward<Wait>(wait), *required_tick);
    return tracker.CompleteThrough(*required_tick);
}

} // namespace VideoCore
