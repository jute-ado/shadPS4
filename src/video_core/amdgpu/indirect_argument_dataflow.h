// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <limits>

#include "common/types.h"

namespace AmdGpu {

struct IndirectArgumentObservation {
    VAddr address{};
    u32 size{};
    u64 content_hash{};
    bool gpu_modified{};
    bool content_hash_valid{};
    bool zero_command{};
};

struct IndirectArgumentFrameReport {
    u32 indirect_draws{};
    u32 gpu_modified_draws{};
    u32 cpu_visible_draws{};
    u32 first_observed_ranges{};
    u32 content_changes{};
    u32 zero_commands{};
    u32 query_dumps{};
    u32 query_overlap_pairs{};
    u32 truncated_indirect_ranges{};
    u32 truncated_query_ranges{};
    u32 truncated_history_ranges{};
};

class IndirectArgumentDataflowTracker {
public:
    static constexpr u32 MaxFrameRanges = 128;
    static constexpr u32 MaxHistoryRanges = 256;

    void ObserveIndirect(const IndirectArgumentObservation& observation) {
        ++report.indirect_draws;
        report.gpu_modified_draws += observation.gpu_modified;
        report.zero_commands += observation.zero_command;

        const Range range{observation.address, observation.size};
        if (range.size != 0) {
            for (u32 i = 0; i < query_range_count; ++i) {
                report.query_overlap_pairs += Overlaps(range, query_ranges[i]);
            }
            if (indirect_range_count < indirect_ranges.size()) {
                indirect_ranges[indirect_range_count++] = range;
            } else {
                ++report.truncated_indirect_ranges;
            }
        }

        if (!observation.content_hash_valid) {
            return;
        }
        ++report.cpu_visible_draws;
        for (u32 i = 0; i < history_count; ++i) {
            auto& entry = history[i];
            if (entry.range.address == range.address && entry.range.size == range.size) {
                if (entry.content_hash != observation.content_hash) {
                    ++report.content_changes;
                    entry.content_hash = observation.content_hash;
                }
                return;
            }
        }
        if (history_count < history.size()) {
            history[history_count++] = {
                .range = range,
                .content_hash = observation.content_hash,
            };
            ++report.first_observed_ranges;
        } else {
            ++report.truncated_history_ranges;
        }
    }

    void ObserveQueryDump(VAddr address, u32 stride_bytes, u32 instances) {
        ++report.query_dumps;
        for (u32 instance = 0; instance < instances; ++instance) {
            const VAddr offset = static_cast<VAddr>(instance) * stride_bytes;
            const VAddr lane_address = address > std::numeric_limits<VAddr>::max() - offset
                                           ? std::numeric_limits<VAddr>::max()
                                           : address + offset;
            const Range range{lane_address, sizeof(u64)};
            for (u32 i = 0; i < indirect_range_count; ++i) {
                report.query_overlap_pairs += Overlaps(range, indirect_ranges[i]);
            }
            if (query_range_count < query_ranges.size()) {
                query_ranges[query_range_count++] = range;
            } else {
                ++report.truncated_query_ranges;
            }
        }
    }

    IndirectArgumentFrameReport TakeFrameReport() {
        const auto completed = report;
        report = {};
        indirect_range_count = 0;
        query_range_count = 0;
        return completed;
    }

private:
    struct Range {
        VAddr address{};
        u64 size{};
    };

    struct HistoryEntry {
        Range range{};
        u64 content_hash{};
    };

    static constexpr VAddr End(const Range& range) {
        return range.address > std::numeric_limits<VAddr>::max() - range.size
                   ? std::numeric_limits<VAddr>::max()
                   : range.address + range.size;
    }

    static constexpr bool Overlaps(const Range& left, const Range& right) {
        return left.size != 0 && right.size != 0 && left.address < End(right) &&
               right.address < End(left);
    }

    IndirectArgumentFrameReport report{};
    std::array<Range, MaxFrameRanges> indirect_ranges{};
    std::array<Range, MaxFrameRanges> query_ranges{};
    std::array<HistoryEntry, MaxHistoryRanges> history{};
    u32 indirect_range_count{};
    u32 query_range_count{};
    u32 history_count{};
};

} // namespace AmdGpu
