// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <span>

#include "common/types.h"

namespace AmdGpu {

struct Gfx7CondExecPacket {
    u64 address{};
    u32 exec_count{};
    bool reserved_control_nonzero{};
    bool legacy_count_matches{};
};

constexpr std::optional<Gfx7CondExecPacket> DecodeGfx7CondExec(std::span<const u32> body) {
    if (body.size() != 4) {
        return std::nullopt;
    }
    constexpr u32 ExecCountMask = (1U << 14) - 1;
    const u64 address = static_cast<u64>(body[1]) << 32 | (body[0] & ~u64{3});
    const u32 exec_count = body[3] & ExecCountMask;
    return Gfx7CondExecPacket{
        .address = address,
        .exec_count = exec_count,
        .reserved_control_nonzero = body[2] != 0,
        .legacy_count_matches = (body[2] & ExecCountMask) == exec_count,
    };
}

struct CondExecDiagnosticReport {
    u32 total_packets{};
    u32 gfx7_packets{};
    u32 decode_rejections{};
    u32 nonzero_exec_counts{};
    u32 retained_records{};
    u32 truncated_records{};
    u32 false_conditions{};
    u32 full_word_nonzero_low_byte_zero{};
    u32 gpu_modified_conditions{};
    u32 layout_mismatches{};
    u32 query_dumps{};
    u32 query_condition_overlaps{};
    u32 truncated_query_ranges{};
    u32 predicated_packets{};
    u32 set_predication_packets{};
    u32 reserved_control_nonzero{};

    auto operator<=>(const CondExecDiagnosticReport&) const = default;
};

class CondExecFrameChangeTracker {
public:
    static constexpr u32 MaxRecords = 64;

    bool ShouldReport(const CondExecDiagnosticReport& report) {
        if (has_last_report && report == last_report) {
            return false;
        }
        last_report = report;
        has_last_report = true;
        if (reported_records < MaxRecords) {
            ++reported_records;
            return true;
        }
        ++truncated_records;
        return false;
    }

    u32 ReportedRecords() const {
        return reported_records;
    }

    u32 TruncatedRecords() const {
        return truncated_records;
    }

private:
    CondExecDiagnosticReport last_report{};
    u32 reported_records{};
    u32 truncated_records{};
    bool has_last_report{};
};

class CondExecDiagnostic {
public:
    static constexpr u32 MaxRecords = 32;
    static constexpr u32 MaxQueryRanges = 64;
    static constexpr u32 MaxQueryLanesPerDump = 16;

    void ObserveRejectedPacket(std::size_t) {
        ++report.total_packets;
        ++report.decode_rejections;
    }

    void Observe(const Gfx7CondExecPacket& packet, u32 condition_word,
                 bool condition_gpu_modified = false) {
        ++report.total_packets;
        ++report.gfx7_packets;
        report.nonzero_exec_counts += packet.exec_count != 0;
        report.false_conditions += condition_word == 0;
        report.full_word_nonzero_low_byte_zero +=
            condition_word != 0 && (condition_word & 0xffU) == 0;
        report.gpu_modified_conditions += condition_gpu_modified;
        report.layout_mismatches += !packet.legacy_count_matches;
        report.reserved_control_nonzero += packet.reserved_control_nonzero;
        bool query_overlap{};
        for (u32 i = 0; i < query_range_count && !query_overlap; ++i) {
            query_overlap = query_ranges[i].Contains(packet.address);
        }
        report.query_condition_overlaps += query_overlap;
        if (report.retained_records < MaxRecords) {
            condition_addresses[report.retained_records] = packet.address;
            condition_query_overlaps[report.retained_records] = query_overlap;
            ++report.retained_records;
        } else {
            ++report.truncated_records;
        }
    }

    void ObserveQueryDump(u64 address, u32 stride_bytes, u32 instances) {
        ++report.query_dumps;
        if (instances == 0 || stride_bytes < sizeof(u64)) {
            ++report.truncated_query_ranges;
            return;
        }
        const u32 retained_lanes = std::min(instances, MaxQueryLanesPerDump);
        report.truncated_query_ranges += instances - retained_lanes;
        for (u32 lane = 0; lane < retained_lanes; ++lane) {
            const u64 lane_offset = static_cast<u64>(stride_bytes) * lane;
            if (address > std::numeric_limits<u64>::max() - lane_offset ||
                address + lane_offset > std::numeric_limits<u64>::max() - sizeof(u64)) {
                ++report.truncated_query_ranges;
                continue;
            }
            const Range range{.begin = address + lane_offset,
                              .end = address + lane_offset + sizeof(u64)};
            for (u32 i = 0; i < report.retained_records; ++i) {
                if (!condition_query_overlaps[i] && range.Contains(condition_addresses[i])) {
                    condition_query_overlaps[i] = true;
                    ++report.query_condition_overlaps;
                }
            }

            bool known_range{};
            for (u32 i = 0; i < query_range_count; ++i) {
                known_range |= query_ranges[i] == range;
            }
            if (known_range) {
                continue;
            }
            if (query_range_count < query_ranges.size()) {
                query_ranges[query_range_count++] = range;
            } else {
                ++report.truncated_query_ranges;
            }
        }
    }

    void ObservePredicatedPacket() {
        ++report.predicated_packets;
    }

    void ObserveSetPredication() {
        ++report.set_predication_packets;
    }

    const CondExecDiagnosticReport& Report() const {
        return report;
    }

    CondExecDiagnosticReport TakeFrameReport() {
        const auto completed = report;
        report = {};
        condition_addresses = {};
        condition_query_overlaps = {};
        return completed;
    }

    void ResetQueryHistory() {
        query_ranges = {};
        query_range_count = 0;
    }

private:
    struct Range {
        u64 begin{};
        u64 end{};

        bool Contains(u64 address) const {
            return address >= begin && address < end;
        }

        auto operator<=>(const Range&) const = default;
    };

    CondExecDiagnosticReport report{};
    std::array<u64, MaxRecords> condition_addresses{};
    std::array<bool, MaxRecords> condition_query_overlaps{};
    std::array<Range, MaxQueryRanges> query_ranges{};
    u32 query_range_count{};
};

} // namespace AmdGpu
