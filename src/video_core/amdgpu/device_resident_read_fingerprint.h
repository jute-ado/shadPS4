// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <optional>

#include "common/types.h"

namespace AmdGpu {

struct DeviceResidentReadObservation {
    u32 source_buffer_id{};
    u64 semantic_identity{};
    u64 source_offset{};
    u64 source_buffer_size{};
    u64 bound_size{};
    u64 write_serial{};
    bool device_local{};
    bool gpu_modified{};
    bool shader_read_only{};
};

struct DeviceResidentReadSample {
    u64 source_offset{};
    u32 destination_offset{};
    u32 size{};
};

struct DeviceResidentReadRange {
    u32 source_buffer_id{};
    u32 stable_identity{};
    u64 semantic_identity{};
    u64 source_offset{};
    u64 bound_size{};
    u64 write_serial{};
    std::array<DeviceResidentReadSample, 3> samples{};
    u32 sample_count{};
    u32 read_references{};
    bool multi_version{};
    bool source_unavailable{};
};

struct DeviceResidentReadCommitResult {
    std::array<u32, 256> source_buffer_ids{};
    u32 source_count{};
};

struct DeviceResidentReadSemanticOrdinal {
    u32 draw{};
    u32 stage{};
    u32 binding{};
};

struct DeviceResidentReadFramePlan {
    std::array<DeviceResidentReadRange, 256> ranges{};
    u32 range_count{};
    u32 observations{};
    u32 sample_bytes{};
    u32 depth_draws{};
    u32 excluded_non_depth_draws{};
    u32 rejected_non_device_local{};
    u32 rejected_not_gpu_modified{};
    u32 rejected_writable{};
    u32 rejected_write_alias{};
    u32 rejected_out_of_bounds{};
    u32 truncated_observations{};
    u32 truncated_ranges{};
    u32 truncated_history{};
    u32 pin_failures{};
};

class DeviceResidentReadFingerprintPlanner {
public:
    static constexpr u32 SampleBytes = 32;
    static constexpr u32 MaxRangesPerFrame = 256;
    static constexpr u32 MaxBytesPerFrame = MaxRangesPerFrame * SampleBytes * 3;
    static constexpr u32 MaxReportFrames = 2048;
    static constexpr u32 MaxHistoryRanges = 8192;
    static constexpr u32 MaxObservationsPerDraw = 512;

    [[nodiscard]] static constexpr u64 RequiredWindowBytes(u64 atom_size) noexcept {
        if (atom_size == 0) {
            return 0;
        }
        const u64 padded = ((MaxBytesPerFrame + atom_size - 1) / atom_size) * atom_size;
        return MaxReportFrames * padded;
    }

    [[nodiscard]] static constexpr bool ShouldCollect(u64 sequence, u64 start, u64 end) noexcept {
        return sequence >= start && sequence < end;
    }

    [[nodiscard]] static constexpr bool ShouldObserveDraw(u32 draw, u32 minimum_draw) noexcept {
        return draw >= minimum_draw;
    }

    [[nodiscard]] static constexpr DeviceResidentReadSemanticOrdinal DecodeSemanticIdentity(
        u64 identity) noexcept {
        return {
            .draw = static_cast<u32>(identity >> 32),
            .stage = static_cast<u32>((identity >> 24) & 0xff),
            .binding = static_cast<u32>(identity & 0xffffff),
        };
    }

    void BeginDraw() noexcept {
        AbortDraw();
        draw_active = true;
    }

    void Observe(const DeviceResidentReadObservation& observation) noexcept {
        if (!draw_active) {
            return;
        }
        if (!observation.device_local) {
            ++plan.rejected_non_device_local;
            return;
        }
        if (!observation.shader_read_only) {
            ++plan.rejected_writable;
            if (writable_count < writable_sources.size()) {
                writable_sources[writable_count++] = observation.source_buffer_id;
            } else {
                ++plan.truncated_observations;
            }
            return;
        }
        if (!observation.gpu_modified) {
            ++plan.rejected_not_gpu_modified;
            return;
        }
        ++plan.observations;
        if (scratch_count >= scratch.size()) {
            ++plan.truncated_observations;
            return;
        }
        scratch[scratch_count++] = observation;
    }

    [[nodiscard]] DeviceResidentReadCommitResult CommitDraw(bool has_depth) noexcept {
        DeviceResidentReadCommitResult result{};
        if (!draw_active) {
            return result;
        }
        if (!has_depth) {
            ++plan.excluded_non_depth_draws;
            AbortDraw();
            return result;
        }
        ++plan.depth_draws;
        for (u32 i = 0; i < scratch_count; ++i) {
            const auto& observation = scratch[i];
            const bool aliases_write =
                std::find(writable_sources.begin(), writable_sources.begin() + writable_count,
                          observation.source_buffer_id) !=
                writable_sources.begin() + writable_count;
            if (aliases_write) {
                ++plan.rejected_write_alias;
                continue;
            }
            if (!AddRange(observation)) {
                continue;
            }
            const auto end = result.source_buffer_ids.begin() + result.source_count;
            if (std::find(result.source_buffer_ids.begin(), end, observation.source_buffer_id) ==
                end) {
                result.source_buffer_ids[result.source_count++] = observation.source_buffer_id;
            }
        }
        AbortDraw();
        return result;
    }

    void AbortDraw() noexcept {
        scratch_count = 0;
        writable_count = 0;
        draw_active = false;
    }

    void RecordPinFailure(u32 source_buffer_id) noexcept {
        ++plan.pin_failures;
        for (u32 i = 0; i < plan.range_count; ++i) {
            if (plan.ranges[i].source_buffer_id == source_buffer_id) {
                plan.ranges[i].source_unavailable = true;
            }
        }
    }

    [[nodiscard]] DeviceResidentReadFramePlan TakeFramePlan() noexcept {
        AbortDraw();
        const auto completed = plan;
        plan = {};
        return completed;
    }

private:
    [[nodiscard]] bool AddRange(const DeviceResidentReadObservation& observation) noexcept {
        if (observation.bound_size < 4 ||
            observation.source_offset > observation.source_buffer_size ||
            observation.bound_size > observation.source_buffer_size - observation.source_offset) {
            ++plan.rejected_out_of_bounds;
            return false;
        }
        for (u32 i = 0; i < plan.range_count; ++i) {
            auto& existing = plan.ranges[i];
            if (existing.source_buffer_id == observation.source_buffer_id &&
                existing.source_offset == observation.source_offset &&
                existing.bound_size == observation.bound_size) {
                ++existing.read_references;
                existing.multi_version |= existing.write_serial != observation.write_serial;
                return true;
            }
        }
        if (plan.range_count >= MaxRangesPerFrame) {
            ++plan.truncated_ranges;
            return false;
        }

        const auto stable_identity = FindOrCreateIdentity(observation.semantic_identity);
        if (!stable_identity.has_value()) {
            ++plan.truncated_history;
            return false;
        }

        DeviceResidentReadRange range{
            .source_buffer_id = observation.source_buffer_id,
            .stable_identity = *stable_identity,
            .semantic_identity = observation.semantic_identity,
            .source_offset = observation.source_offset,
            .bound_size = observation.bound_size,
            .write_serial = observation.write_serial,
            .read_references = 1,
        };
        const u64 copyable_size = observation.bound_size & ~u64{3};
        if (copyable_size < 4) {
            ++plan.rejected_out_of_bounds;
            return false;
        }
        if (copyable_size <= SampleBytes * 3) {
            AddSample(range, observation.source_offset, static_cast<u32>(copyable_size));
        } else {
            const u64 first = observation.source_offset;
            const u64 middle =
                (observation.source_offset + (copyable_size - SampleBytes) / 2) & ~u64{3};
            const u64 last = observation.source_offset + copyable_size - SampleBytes;
            AddSample(range, first, SampleBytes);
            AddSample(range, middle, SampleBytes);
            AddSample(range, last, SampleBytes);
        }
        plan.ranges[plan.range_count++] = range;
        return true;
    }

    void AddSample(DeviceResidentReadRange& range, u64 source_offset, u32 size) noexcept {
        for (u32 i = 0; i < range.sample_count; ++i) {
            if (range.samples[i].source_offset == source_offset && range.samples[i].size == size) {
                return;
            }
        }
        if (range.sample_count >= range.samples.size() ||
            size > MaxBytesPerFrame - plan.sample_bytes) {
            return;
        }
        range.samples[range.sample_count++] = {
            .source_offset = source_offset,
            .destination_offset = plan.sample_bytes,
            .size = size,
        };
        plan.sample_bytes += size;
    }

    [[nodiscard]] std::optional<u32> FindOrCreateIdentity(u64 semantic_identity) noexcept {
        for (u32 i = 0; i < history_count; ++i) {
            if (history[i] == semantic_identity) {
                return i;
            }
        }
        if (history_count >= history.size()) {
            return std::nullopt;
        }
        history[history_count] = semantic_identity;
        return history_count++;
    }

    DeviceResidentReadFramePlan plan{};
    std::array<DeviceResidentReadObservation, MaxObservationsPerDraw> scratch{};
    std::array<u32, MaxObservationsPerDraw> writable_sources{};
    std::array<u64, MaxHistoryRanges> history{};
    u32 scratch_count{};
    u32 writable_count{};
    u32 history_count{};
    bool draw_active{};
};

struct DeviceResidentReadChange {
    bool first_observation{};
    bool unchanged{};
    bool changed{};
    bool exact_aba_return{};
};

class DeviceResidentReadReducer {
public:
    DeviceResidentReadChange Observe(u32 stable_identity, u64 fingerprint,
                                     u64 sequence = std::numeric_limits<u64>::max()) noexcept {
        for (u32 i = 0; i < count; ++i) {
            auto& entry = entries[i];
            if (entry.stable_identity != stable_identity) {
                continue;
            }
            if (sequence == std::numeric_limits<u64>::max()) {
                sequence = entry.last_sequence + 1;
            }
            const bool contiguous = entry.has_previous && sequence == entry.last_sequence + 1;
            DeviceResidentReadChange result{
                .unchanged = contiguous && entry.previous == fingerprint,
                .changed = contiguous && entry.previous != fingerprint,
                .exact_aba_return = contiguous && entry.has_previous_previous &&
                                    entry.previous_previous_sequence + 2 == sequence &&
                                    entry.previous_previous == fingerprint &&
                                    entry.previous != fingerprint,
            };
            entry.previous_previous = entry.previous;
            entry.previous_previous_sequence = entry.last_sequence;
            entry.has_previous_previous = entry.has_previous;
            entry.previous = fingerprint;
            entry.last_sequence = sequence;
            entry.has_previous = true;
            return result;
        }
        if (count < entries.size()) {
            entries[count++] = {
                .stable_identity = stable_identity,
                .previous = fingerprint,
                .last_sequence = sequence == std::numeric_limits<u64>::max() ? 0 : sequence,
                .has_previous = true,
            };
        }
        return {.first_observation = true};
    }

private:
    struct Entry {
        u32 stable_identity{};
        u64 previous_previous{};
        u64 previous{};
        u64 previous_previous_sequence{};
        u64 last_sequence{};
        bool has_previous{};
        bool has_previous_previous{};
    };
    std::array<Entry, DeviceResidentReadFingerprintPlanner::MaxHistoryRanges> entries{};
    u32 count{};
};

} // namespace AmdGpu
