// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace AmdGpu {

constexpr std::uint64_t GpuCommandWorkReportIntervalNanoseconds = 10'000'000'000ULL;

constexpr bool GpuCommandWorkTimingRequested(const char* value) {
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

enum class GpuCommandWorkCategory : std::size_t {
    CommandCallback,
    Resume,
    Draw,
    Dispatch,
    Transfer,
    Download,
    Sync,
    Submit,
    Wait,
    Count,
};

struct GpuCommandWorkCategoryStats {
    std::uint64_t calls{};
    std::uint64_t nanoseconds{};
};

struct GpuCommandWorkSnapshot {
    std::uint64_t interval_nanoseconds{};
    std::uint64_t packet_count{};
    std::uint64_t packet_dwords{};
    std::array<GpuCommandWorkCategoryStats, static_cast<std::size_t>(GpuCommandWorkCategory::Count)>
        categories{};

    const GpuCommandWorkCategoryStats& At(const GpuCommandWorkCategory category) const {
        return categories[static_cast<std::size_t>(category)];
    }

    std::uint64_t UnclassifiedResumeNanoseconds() const {
        const auto classified = At(GpuCommandWorkCategory::Draw).nanoseconds +
                                At(GpuCommandWorkCategory::Dispatch).nanoseconds +
                                At(GpuCommandWorkCategory::Transfer).nanoseconds +
                                At(GpuCommandWorkCategory::Download).nanoseconds +
                                At(GpuCommandWorkCategory::Sync).nanoseconds +
                                At(GpuCommandWorkCategory::Wait).nanoseconds;
        const auto resume = At(GpuCommandWorkCategory::Resume).nanoseconds;
        return resume > classified ? resume - classified : 0;
    }
};

class GpuCommandWorkTiming {
public:
    explicit GpuCommandWorkTiming(const std::uint64_t interval_start_nanoseconds)
        : interval_start_nanoseconds{interval_start_nanoseconds} {}

    void Record(const GpuCommandWorkCategory category, const std::uint64_t nanoseconds) {
        auto& stats = categories[static_cast<std::size_t>(category)];
        ++stats.calls;
        stats.nanoseconds += nanoseconds;
    }

    void RecordPacket(const std::uint64_t dwords) {
        ++packet_count;
        packet_dwords += dwords;
    }

    bool ShouldReport(const std::uint64_t now_nanoseconds) const {
        return now_nanoseconds >= interval_start_nanoseconds &&
               now_nanoseconds - interval_start_nanoseconds >=
                   GpuCommandWorkReportIntervalNanoseconds;
    }

    GpuCommandWorkSnapshot TakeSnapshot(const std::uint64_t now_nanoseconds) {
        GpuCommandWorkSnapshot snapshot{
            .interval_nanoseconds = now_nanoseconds >= interval_start_nanoseconds
                                        ? now_nanoseconds - interval_start_nanoseconds
                                        : 0,
            .packet_count = packet_count,
            .packet_dwords = packet_dwords,
            .categories = categories,
        };
        interval_start_nanoseconds = now_nanoseconds;
        packet_count = 0;
        packet_dwords = 0;
        categories = {};
        return snapshot;
    }

private:
    std::uint64_t interval_start_nanoseconds{};
    std::uint64_t packet_count{};
    std::uint64_t packet_dwords{};
    std::array<GpuCommandWorkCategoryStats, static_cast<std::size_t>(GpuCommandWorkCategory::Count)>
        categories{};
};

} // namespace AmdGpu
