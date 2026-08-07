// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <limits>
#include <optional>
#include <span>

#include "common/types.h"

namespace AmdGpu {

class DiagnosticReadbackPin {
public:
    void Acquire() {
        ++count;
    }

    bool Release() {
        if (count == 0) {
            return false;
        }
        --count;
        const bool delete_now = count == 0 && delete_pending;
        delete_pending &= !delete_now;
        return delete_now;
    }

    bool RequestDelete() {
        if (count == 0) {
            return true;
        }
        delete_pending = true;
        return false;
    }

    bool IsPinned() const {
        return count != 0;
    }

    bool IsDeletePending() const {
        return delete_pending;
    }

private:
    u32 count{};
    bool delete_pending{};
};

struct IndirectArgumentReadbackObservation {
    u64 source_token{};
    u64 range_identity{};
    u64 source_offset{};
    u32 stride{};
    u32 max_count{};
    bool indexed{};
    bool gpu_modified{};
};

struct IndirectArgumentReadbackRecord {
    u64 source_token{};
    u64 source_offset{};
    u32 destination_offset{};
    u32 stable_identity{};
};

struct IndirectArgumentReadbackPlan {
    std::array<IndirectArgumentReadbackRecord, 16> records{};
    u32 record_count{};
    u32 batch_count{};
    u32 rejected_cpu_visible{};
    u32 rejected_nonindexed{};
    u32 rejected_stride{};
    u32 rejected_overflow{};
    u32 truncated_records{};
    u32 truncated_history{};
};

class IndirectArgumentReadbackPlanner {
public:
    static constexpr u32 CommandBytes = 20;
    static constexpr u32 MaxRecordsPerFrame = 16;
    static constexpr u32 MaxHistoryRanges = 256;
    static constexpr u32 MaxReportFrames = 5000;
    static constexpr u32 DownloadRingBytes = 2 * 1024 * 1024;
    static constexpr u32 RequiredWindowBytes =
        MaxReportFrames * MaxRecordsPerFrame * CommandBytes;

    void Observe(const IndirectArgumentReadbackObservation& observation) {
        if (!observation.gpu_modified) {
            ++plan.rejected_cpu_visible;
            return;
        }
        if (!observation.indexed) {
            ++plan.rejected_nonindexed;
            return;
        }
        if (observation.stride != CommandBytes) {
            ++plan.rejected_stride;
            return;
        }

        for (u32 command = 0; command < observation.max_count; ++command) {
            const u64 command_offset = static_cast<u64>(command) * observation.stride;
            if (observation.source_offset > std::numeric_limits<u64>::max() - command_offset ||
                observation.source_offset + command_offset >
                    std::numeric_limits<u64>::max() - CommandBytes ||
                observation.range_identity > std::numeric_limits<u64>::max() - command_offset) {
                ++plan.rejected_overflow;
                continue;
            }
            if (plan.record_count >= MaxRecordsPerFrame) {
                ++plan.truncated_records;
                continue;
            }

            const u64 source_offset = observation.source_offset + command_offset;
            const u64 range_identity = observation.range_identity + command_offset;
            const auto stable_identity = FindOrCreateIdentity(range_identity);
            if (!stable_identity.has_value()) {
                ++plan.truncated_history;
                continue;
            }

            bool source_seen{};
            for (u32 i = 0; i < plan.record_count; ++i) {
                source_seen |= plan.records[i].source_token == observation.source_token;
            }
            if (!source_seen) {
                ++plan.batch_count;
            }
            plan.records[plan.record_count] = {
                .source_token = observation.source_token,
                .source_offset = source_offset,
                .destination_offset = plan.record_count * CommandBytes,
                .stable_identity = *stable_identity,
            };
            ++plan.record_count;
        }
    }

    IndirectArgumentReadbackPlan TakeFramePlan() {
        const auto completed = plan;
        plan = {};
        return completed;
    }

private:
    std::optional<u32> FindOrCreateIdentity(u64 range_identity) {
        for (u32 i = 0; i < history_count; ++i) {
            if (history[i] == range_identity) {
                return i;
            }
        }
        if (history_count >= history.size()) {
            return std::nullopt;
        }
        history[history_count] = range_identity;
        return history_count++;
    }

    IndirectArgumentReadbackPlan plan{};
    std::array<u64, MaxHistoryRanges> history{};
    u32 history_count{};
};

struct IndirectReadbackReservation {
    u32 offset{};
    u32 next_offset{};
};

constexpr std::optional<IndirectReadbackReservation> TryReserveIndirectReadback(
    u32 current_offset, u32 size, bool busy) {
    if (busy) {
        return std::nullopt;
    }
    constexpr u32 Alignment = 4;
    const u32 aligned = (current_offset + Alignment - 1) & ~(Alignment - 1);
    if (aligned < current_offset || size > IndirectArgumentReadbackPlanner::DownloadRingBytes ||
        aligned > IndirectArgumentReadbackPlanner::DownloadRingBytes - size) {
        return std::nullopt;
    }
    return IndirectReadbackReservation{.offset = aligned, .next_offset = aligned + size};
}

struct IndirectArgumentReadbackChange {
    u32 changed_field_mask{};
    bool first_observation{};
    bool zero_index_count{};
    bool zero_instance_count{};
};

class IndirectArgumentReadbackReducer {
public:
    IndirectArgumentReadbackChange Observe(u32 stable_identity, std::span<const u32, 5> command) {
        IndirectArgumentReadbackChange result{
            .zero_index_count = command[0] == 0,
            .zero_instance_count = command[1] == 0,
        };
        for (u32 i = 0; i < count; ++i) {
            auto& entry = entries[i];
            if (entry.stable_identity != stable_identity) {
                continue;
            }
            for (u32 field = 0; field < command.size(); ++field) {
                if (entry.command[field] != command[field]) {
                    result.changed_field_mask |= 1U << field;
                }
            }
            std::ranges::copy(command, entry.command.begin());
            return result;
        }
        if (count < entries.size()) {
            entries[count].stable_identity = stable_identity;
            std::ranges::copy(command, entries[count].command.begin());
            ++count;
            result.first_observation = true;
        }
        return result;
    }

private:
    struct Entry {
        u32 stable_identity{};
        std::array<u32, 5> command{};
    };

    std::array<Entry, IndirectArgumentReadbackPlanner::MaxHistoryRanges> entries{};
    u32 count{};
};

static_assert(IndirectArgumentReadbackPlanner::RequiredWindowBytes <=
              IndirectArgumentReadbackPlanner::DownloadRingBytes);

} // namespace AmdGpu
