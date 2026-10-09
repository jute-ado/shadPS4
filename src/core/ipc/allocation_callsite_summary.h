// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Core::Ipc {

struct AllocationCallsiteStat {
    std::uintptr_t caller{};
    std::size_t size{};
    std::size_t count{};
    std::size_t total_bytes{};
};

class AllocationCallsiteSummary {
public:
    static constexpr std::size_t Capacity = 16384;

    void Insert(std::uintptr_t caller, std::size_t size) noexcept {
        if (caller == 0 || size == 0) {
            return;
        }
        std::size_t index = ((caller >> 4) ^ (size * 11400714819323198485ull)) & (Capacity - 1);
        for (std::size_t probe = 0; probe < Capacity; ++probe) {
            auto& stat = stats[index];
            if (stat.caller == 0 || (stat.caller == caller && stat.size == size)) {
                stat.caller = caller;
                stat.size = size;
                ++stat.count;
                stat.total_bytes += size;
                return;
            }
            index = (index + 1) & (Capacity - 1);
        }
        ++overflow_count;
        overflow_bytes += size;
    }

    [[nodiscard]] std::vector<AllocationCallsiteStat> TopByTotalBytes() const {
        std::vector<AllocationCallsiteStat> populated;
        populated.reserve(Capacity);
        for (const auto& stat : stats) {
            if (stat.caller != 0) {
                populated.push_back(stat);
            }
        }
        std::ranges::sort(populated, std::greater{}, &AllocationCallsiteStat::total_bytes);
        return populated;
    }

    [[nodiscard]] std::size_t OverflowCount() const noexcept { return overflow_count; }
    [[nodiscard]] std::size_t OverflowBytes() const noexcept { return overflow_bytes; }

private:
    static_assert((Capacity & (Capacity - 1)) == 0);
    std::array<AllocationCallsiteStat, Capacity> stats{};
    std::size_t overflow_count{};
    std::size_t overflow_bytes{};
};

template <std::size_t Capacity>
class SampledAllocationTracker {
public:
    static_assert((Capacity & (Capacity - 1)) == 0);

    bool Add(void* pointer, std::uintptr_t caller, std::size_t size) noexcept {
        if (pointer == nullptr || pointer == Tombstone()) {
            return false;
        }
        std::size_t index = (reinterpret_cast<std::uintptr_t>(pointer) >> 4) & (Capacity - 1);
        std::size_t tombstone_index = Capacity;
        for (std::size_t probe = 0; probe < Capacity; ++probe) {
            auto& entry = entries[index];
            if (entry.pointer == nullptr) {
                auto& target = entries[tombstone_index == Capacity ? index : tombstone_index];
                target = {pointer, caller, size};
                return true;
            }
            if (entry.pointer == Tombstone() && tombstone_index == Capacity) {
                tombstone_index = index;
            }
            index = (index + 1) & (Capacity - 1);
        }
        return false;
    }

    bool Remove(void* pointer) noexcept {
        if (pointer == nullptr) {
            return false;
        }
        std::size_t index = (reinterpret_cast<std::uintptr_t>(pointer) >> 4) & (Capacity - 1);
        for (std::size_t probe = 0; probe < Capacity; ++probe) {
            auto& entry = entries[index];
            if (entry.pointer == nullptr) {
                return false;
            }
            if (entry.pointer == pointer) {
                entry = {Tombstone(), 0, 0};
                return true;
            }
            index = (index + 1) & (Capacity - 1);
        }
        return false;
    }

    [[nodiscard]] AllocationCallsiteSummary Summarize() const noexcept {
        AllocationCallsiteSummary summary;
        for (const auto& entry : entries) {
            if (entry.pointer != nullptr && entry.pointer != Tombstone()) {
                summary.Insert(entry.caller, entry.size);
            }
        }
        return summary;
    }

private:
    struct Entry {
        void* pointer{};
        std::uintptr_t caller{};
        std::size_t size{};
    };

    static void* Tombstone() noexcept { return reinterpret_cast<void*>(std::uintptr_t{1}); }
    std::array<Entry, Capacity> entries{};
};

} // namespace Core::Ipc
