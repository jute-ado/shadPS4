// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include "common/types.h"

namespace AmdGpu {

[[nodiscard]] constexpr bool ShouldInspectLodStats(const u32 request_index) {
    return request_index < 8 || std::has_single_bit(request_index);
}

struct LodStatsRequest {
    VAddr address;
    u32 size;
    u8 reset_count;
    bool force_reset;
    bool report_and_reset;
    u8 reporting_interval;
};

struct LodStatsReadWatch {
    VAddr address;
    u64 size;
    VAddr page_address;
    u64 page_size;

    [[nodiscard]] constexpr bool Contains(const VAddr candidate) const {
        return candidate >= address && candidate - address < size;
    }
};

[[nodiscard]] constexpr LodStatsReadWatch MakeLodStatsReadWatch(
    const VAddr address, const u64 size, const u64 page_size) {
    const VAddr page_address = address & ~(page_size - 1);
    const VAddr end = address + size;
    const VAddr page_end = (end + page_size - 1) & ~(page_size - 1);
    return {
        .address = address,
        .size = size,
        .page_address = page_address,
        .page_size = page_end - page_address,
    };
}

[[nodiscard]] constexpr LodStatsRequest DecodeLodStatsRequest(
    std::span<const u32, 3> payload) {
    const u32 control = payload[2];
    return {
        .address = (payload[1] & 0xffffffc0u) | (u64(control & 0xffu) << 32u),
        .size = payload[0],
        .reset_count = static_cast<u8>((control >> 10u) & 0xffu),
        .force_reset = ((control >> 18u) & 1u) != 0,
        .report_and_reset = ((control >> 19u) & 1u) != 0,
        .reporting_interval = static_cast<u8>((control >> 2u) & 0xffu),
    };
}

inline constexpr u32 MaxLodStatsReportSize = 0x1020;

[[nodiscard]] constexpr bool IsLodStatsReportSizeSupported(const u32 size) {
    return size >= sizeof(u32) && size <= MaxLodStatsReportSize && size % sizeof(u32) == 0;
}

inline void InitializeLodStatsReport(std::span<std::byte> output) {
    std::ranges::fill(output, std::byte{});
    if (output.size() >= sizeof(u32)) {
        constexpr u32 ReportComplete = 1;
        std::memcpy(output.data(), &ReportComplete, sizeof(ReportComplete));
    }
}

struct LodStatsReportSummary {
    u32 nonzero_dwords{};
    u32 first_nonzero_dword{std::numeric_limits<u32>::max()};
    u32 first_nonzero_value{};
};

[[nodiscard]] inline LodStatsReportSummary SummarizeLodStatsReport(std::span<const u32> report) {
    LodStatsReportSummary summary;
    for (u32 index = 0; index < report.size(); ++index) {
        if (report[index] == 0) {
            continue;
        }
        if (summary.nonzero_dwords++ == 0) {
            summary.first_nonzero_dword = index;
            summary.first_nonzero_value = report[index];
        }
    }
    return summary;
}

} // namespace AmdGpu
