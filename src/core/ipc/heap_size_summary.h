// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Core::Ipc {

struct HeapSizeStat {
    std::size_t size{};
    std::size_t count{};
    std::size_t total_bytes{};
    std::uintptr_t sample_address{};
    std::uintptr_t last_sample_address{};
};

class HeapSizeSummary {
public:
    static constexpr std::size_t Capacity = 8192;

    void Insert(const std::size_t size, const std::uintptr_t sample_address = 0) noexcept {
        if (size == 0) {
            return;
        }
        std::size_t index = (size * 11400714819323198485ull) & (Capacity - 1);
        for (std::size_t probe = 0; probe < Capacity; ++probe) {
            auto& stat = stats[index];
            if (stat.size == 0 || stat.size == size) {
                stat.size = size;
                if (stat.sample_address == 0) {
                    stat.sample_address = sample_address;
                }
                stat.last_sample_address = sample_address;
                ++stat.count;
                stat.total_bytes += size;
                return;
            }
            index = (index + 1) & (Capacity - 1);
        }
        ++overflow_count;
        overflow_bytes += size;
    }

    [[nodiscard]] std::vector<HeapSizeStat> TopByTotalBytes() const {
        std::vector<HeapSizeStat> populated;
        populated.reserve(Capacity);
        for (const auto& stat : stats) {
            if (stat.size != 0) {
                populated.push_back(stat);
            }
        }
        std::ranges::sort(populated, std::greater{}, &HeapSizeStat::total_bytes);
        return populated;
    }

    [[nodiscard]] std::size_t OverflowCount() const noexcept {
        return overflow_count;
    }

    [[nodiscard]] std::size_t OverflowBytes() const noexcept {
        return overflow_bytes;
    }

private:
    static_assert((Capacity & (Capacity - 1)) == 0);
    std::array<HeapSizeStat, Capacity> stats{};
    std::size_t overflow_count{};
    std::size_t overflow_bytes{};
};

} // namespace Core::Ipc
