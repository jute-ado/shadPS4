// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "common/types.h"

namespace AmdGpu {

enum class InputAssemblySourceKind : u8 {
    Vertex,
    Index,
};

enum class InputAssemblyAuthority : u8 {
    Unknown,
    CpuAuthoritative,
    GpuAuthoritative,
};

struct InputAssemblyBufferToken {
    u32 slot{};
    u32 generation{};

    auto operator<=>(const InputAssemblyBufferToken&) const = default;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return generation != 0;
    }
};

struct InputAssemblySemanticOrdinal {
    u32 draw{};
    InputAssemblySourceKind kind{};
    u32 binding{};

    auto operator<=>(const InputAssemblySemanticOrdinal&) const = default;
};

struct InputAssemblyObservation {
    InputAssemblySemanticOrdinal semantic{};
    InputAssemblyBufferToken source{};
    u64 source_offset{};
    u64 source_size{};
    u64 size{};
    u64 write_serial{};
    InputAssemblyAuthority authority{InputAssemblyAuthority::Unknown};
};

struct InputAssemblySemanticReference {
    InputAssemblySemanticOrdinal semantic{};
    u32 stable_identity{};
    u64 relative_offset{};
    u64 size{};
};

struct InputAssemblyRange {
    static constexpr u32 MaxSemantics = 64;

    InputAssemblyBufferToken source{};
    u64 source_offset{};
    u64 size{};
    u64 write_serial{};
    InputAssemblyAuthority authority{InputAssemblyAuthority::Unknown};
    std::array<InputAssemblySemanticReference, MaxSemantics> semantics{};
    u32 semantic_count{};
    bool write_serial_ambiguous{};
    bool authority_ambiguous{};
};

struct InputAssemblyLoss {
    u32 zero_size{};
    u32 out_of_bounds{};
    u32 invalid_source{};
    u32 range_capacity{};
    u32 byte_capacity{};
    u32 history_capacity{};
    u32 semantic_capacity{};
    u32 write_serial_ambiguity{};
    u32 authority_ambiguity{};

    [[nodiscard]] constexpr bool Any() const noexcept {
        return zero_size != 0 || out_of_bounds != 0 || invalid_source != 0 || range_capacity != 0 ||
               byte_capacity != 0 || history_capacity != 0 || semantic_capacity != 0 ||
               write_serial_ambiguity != 0 || authority_ambiguity != 0;
    }
};

struct InputAssemblyFramePlan {
    std::vector<InputAssemblyRange> ranges{};
    u64 sequence{};
    u64 byte_count{};
    u32 range_count{};
    InputAssemblyLoss loss{};
    bool complete{true};
};

class InputAssemblyDeviceIntegrityPlanner {
public:
    static constexpr u32 MaxRangesPerFrame = 256;
    static constexpr u64 MaxBytesPerFrame = 24 * 1024;
    static constexpr u32 MaxHistoryIdentities = 8192;

    InputAssemblyDeviceIntegrityPlanner() {
        plan.ranges.reserve(MaxRangesPerFrame);
    }

    void BeginFrame(u64 sequence) {
        plan.ranges.clear();
        plan.sequence = sequence;
        plan.byte_count = 0;
        plan.range_count = 0;
        plan.loss = {};
        plan.complete = true;
        frame_active = true;
    }

    void Observe(const InputAssemblyObservation& observation) noexcept {
        if (!frame_active) {
            return;
        }
        if (!observation.source) {
            Reject(plan.loss.invalid_source);
            return;
        }
        if (observation.size == 0) {
            Reject(plan.loss.zero_size);
            return;
        }
        if (observation.source_offset > observation.source_size ||
            observation.size > observation.source_size - observation.source_offset) {
            Reject(plan.loss.out_of_bounds);
            return;
        }

        const u64 observation_end = observation.source_offset + observation.size;
        std::array<bool, MaxRangesPerFrame> affected{};
        u64 merged_begin = observation.source_offset;
        u64 merged_end = observation_end;
        bool expanded{};
        do {
            expanded = false;
            for (u32 i = 0; i < plan.range_count; ++i) {
                const auto& range = plan.ranges[i];
                if (affected[i] || range.source != observation.source) {
                    continue;
                }
                const u64 range_end = range.source_offset + range.size;
                if (merged_begin <= range_end && range.source_offset <= merged_end) {
                    affected[i] = true;
                    merged_begin = std::min(merged_begin, range.source_offset);
                    merged_end = std::max(merged_end, range_end);
                    expanded = true;
                }
            }
        } while (expanded);

        InputAssemblyRange merged{
            .source = observation.source,
            .source_offset = merged_begin,
            .size = merged_end - merged_begin,
            .write_serial = observation.write_serial,
            .authority = observation.authority,
        };
        u32 affected_count{};
        u64 affected_bytes{};
        u32 first_affected = MaxRangesPerFrame;
        bool inherited_serial_ambiguity{};
        bool inherited_authority_ambiguity{};
        for (u32 i = 0; i < plan.range_count; ++i) {
            if (!affected[i]) {
                continue;
            }
            const auto& range = plan.ranges[i];
            ++affected_count;
            affected_bytes += range.size;
            first_affected = std::min(first_affected, i);
            inherited_serial_ambiguity |= range.write_serial_ambiguous;
            inherited_authority_ambiguity |= range.authority_ambiguous;
            if (merged.write_serial != range.write_serial || range.write_serial_ambiguous) {
                merged.write_serial_ambiguous = true;
            }
            if (merged.authority != range.authority || range.authority_ambiguous) {
                merged.authority = InputAssemblyAuthority::Unknown;
                merged.authority_ambiguous = true;
            }
            for (u32 semantic = 0; semantic < range.semantic_count; ++semantic) {
                const auto& reference = range.semantics[semantic];
                const u64 absolute_offset = range.source_offset + reference.relative_offset;
                if (!AddSemantic(merged, reference.semantic, reference.stable_identity,
                                 absolute_offset - merged_begin, reference.size)) {
                    Reject(plan.loss.semantic_capacity);
                    return;
                }
            }
        }

        const u32 resulting_range_count = plan.range_count - affected_count + 1;
        if (resulting_range_count > MaxRangesPerFrame) {
            Reject(plan.loss.range_capacity);
            return;
        }
        const u64 resulting_bytes = plan.byte_count - affected_bytes + merged.size;
        if (resulting_bytes > MaxBytesPerFrame) {
            Reject(plan.loss.byte_capacity);
            return;
        }

        auto stable_identity = FindIdentity(observation.semantic);
        if (!stable_identity.has_value() && history_count >= history.size()) {
            Reject(plan.loss.history_capacity);
            return;
        }
        const u32 accepted_identity = stable_identity.value_or(history_count);
        if (!AddSemantic(merged, observation.semantic, accepted_identity,
                         observation.source_offset - merged_begin, observation.size)) {
            Reject(plan.loss.semantic_capacity);
            return;
        }
        if (!stable_identity.has_value()) {
            history[history_count++] = observation.semantic;
        }

        if (merged.write_serial_ambiguous && !inherited_serial_ambiguity) {
            Reject(plan.loss.write_serial_ambiguity);
        }
        if (merged.authority_ambiguous && !inherited_authority_ambiguity) {
            Reject(plan.loss.authority_ambiguity);
        }

        if (affected_count == 0) {
            plan.ranges.push_back(merged);
        } else {
            plan.ranges[first_affected] = merged;
            for (u32 i = plan.range_count; i-- > 0;) {
                if (affected[i] && i != first_affected) {
                    plan.ranges.erase(plan.ranges.begin() + i);
                }
            }
        }
        plan.range_count = resulting_range_count;
        plan.byte_count = resulting_bytes;
    }

    [[nodiscard]] const InputAssemblyFramePlan& EndFrame() noexcept {
        frame_active = false;
        plan.complete &= !plan.loss.Any();
        return plan;
    }

private:
    void Reject(u32& counter) noexcept {
        ++counter;
        plan.complete = false;
    }

    [[nodiscard]] std::optional<u32> FindIdentity(
        const InputAssemblySemanticOrdinal& semantic) const noexcept {
        for (u32 i = 0; i < history_count; ++i) {
            if (history[i] == semantic) {
                return i;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] static bool AddSemantic(InputAssemblyRange& range,
                                          const InputAssemblySemanticOrdinal& semantic,
                                          u32 stable_identity, u64 relative_offset,
                                          u64 size) noexcept {
        for (u32 i = 0; i < range.semantic_count; ++i) {
            if (range.semantics[i].semantic == semantic) {
                return true;
            }
        }
        if (range.semantic_count >= range.semantics.size()) {
            return false;
        }
        range.semantics[range.semantic_count++] = {
            .semantic = semantic,
            .stable_identity = stable_identity,
            .relative_offset = relative_offset,
            .size = size,
        };
        return true;
    }

    InputAssemblyFramePlan plan{};
    std::array<InputAssemblySemanticOrdinal, MaxHistoryIdentities> history{};
    u32 history_count{};
    bool frame_active{};
};

struct InputAssemblyRingReservation {
    u64 offset{};
    u64 reserved_size{};
    u64 completion_tick{};

    [[nodiscard]] constexpr bool Overlaps(
        const InputAssemblyRingReservation& other) const noexcept {
        return offset < other.offset + other.reserved_size && other.offset < offset + reserved_size;
    }
};

class InputAssemblyReadbackRing {
public:
    InputAssemblyReadbackRing(u64 capacity, u64 non_coherent_atom_size) noexcept
        : capacity{capacity}, atom_size{non_coherent_atom_size} {}

    [[nodiscard]] std::optional<InputAssemblyRingReservation> TryReserve(
        u64 size, u64 completion_tick) noexcept {
        if (size == 0 || atom_size == 0 || size > std::numeric_limits<u64>::max() - atom_size + 1) {
            ++busy_count;
            return std::nullopt;
        }
        const u64 reserved_size = ((size + atom_size - 1) / atom_size) * atom_size;
        if (reserved_size > capacity || active_count >= active.size()) {
            ++busy_count;
            return std::nullopt;
        }

        for (u64 candidate = 0; candidate <= capacity - reserved_size; candidate += atom_size) {
            const InputAssemblyRingReservation reservation{candidate, reserved_size,
                                                           completion_tick};
            bool overlaps{};
            for (u32 i = 0; i < active_count; ++i) {
                overlaps |= reservation.Overlaps(active[i]);
            }
            if (!overlaps) {
                active[active_count++] = reservation;
                return reservation;
            }
        }
        ++busy_count;
        return std::nullopt;
    }

    void ReleaseCompleted(u64 completed_tick) noexcept {
        u32 retained{};
        for (u32 i = 0; i < active_count; ++i) {
            if (active[i].completion_tick > completed_tick) {
                active[retained++] = active[i];
            }
        }
        active_count = retained;
    }

    [[nodiscard]] u32 BusyCount() const noexcept {
        return busy_count;
    }

private:
    u64 capacity{};
    u64 atom_size{};
    std::array<InputAssemblyRingReservation, 256> active{};
    u32 active_count{};
    u32 busy_count{};
};

struct InputAssemblySnapshot {
    u64 sequence{};
    u32 stable_identity{};
    InputAssemblyBufferToken source{};
    u64 write_serial{};
    InputAssemblyAuthority authority{InputAssemblyAuthority::Unknown};
    std::span<const std::byte> bytes{};
    bool complete{};
};

struct InputAssemblyChange {
    bool first_observation{};
    bool changed{};
    bool exact_aba_return{};
    bool authority_ambiguous{};
    bool write_serial_ambiguous{};
    bool source_generation_ambiguous{};
    bool sequence_gap{};
    bool incomplete{};
    bool baseline_reset{};
    bool empty_snapshot{};
    bool capacity_exceeded{};
};

class InputAssemblyDeviceIntegrityReducer {
public:
    static constexpr u32 MaxEntries = 8192;
    static constexpr u32 MaxSnapshotBytes = 96;

    InputAssemblyDeviceIntegrityReducer() : entries{std::make_unique<Entry[]>(MaxEntries)} {}

    [[nodiscard]] InputAssemblyChange Observe(const InputAssemblySnapshot& snapshot) noexcept {
        if (snapshot.bytes.empty()) {
            return {.empty_snapshot = true};
        }
        if (snapshot.bytes.size() > MaxSnapshotBytes) {
            return {.capacity_exceeded = true};
        }

        Entry* entry{};
        for (u32 i = 0; i < count; ++i) {
            if (entries[i].stable_identity == snapshot.stable_identity) {
                entry = &entries[i];
                break;
            }
        }
        if (entry == nullptr) {
            if (count >= MaxEntries) {
                return {.capacity_exceeded = true};
            }
            entry = &entries[count++];
            entry->stable_identity = snapshot.stable_identity;
        }

        if (!snapshot.complete) {
            ClearBaseline(*entry);
            entry->observed = true;
            return {.incomplete = true};
        }
        if (snapshot.authority != InputAssemblyAuthority::GpuAuthoritative) {
            ClearBaseline(*entry);
            entry->observed = true;
            entry->authority = snapshot.authority;
            entry->source = snapshot.source;
            entry->write_serial = snapshot.write_serial;
            entry->last_sequence = snapshot.sequence;
            return {.authority_ambiguous = true};
        }
        if (!snapshot.source) {
            ClearBaseline(*entry);
            entry->observed = true;
            return {.source_generation_ambiguous = true};
        }
        if (!entry->has_baseline) {
            const bool prior_observation = entry->observed;
            SetBaseline(*entry, snapshot);
            return prior_observation ? InputAssemblyChange{.baseline_reset = true}
                                     : InputAssemblyChange{.first_observation = true};
        }
        if (entry->source != snapshot.source) {
            SetBaseline(*entry, snapshot);
            return {.source_generation_ambiguous = true};
        }
        if (entry->write_serial != snapshot.write_serial) {
            SetBaseline(*entry, snapshot);
            return {.write_serial_ambiguous = true};
        }
        if (entry->authority != snapshot.authority) {
            SetBaseline(*entry, snapshot);
            return {.baseline_reset = true};
        }
        const bool contiguous = entry->last_sequence != std::numeric_limits<u64>::max() &&
                                snapshot.sequence == entry->last_sequence + 1;
        if (!contiguous) {
            SetBaseline(*entry, snapshot);
            return {.sequence_gap = true};
        }

        const auto previous =
            std::span<const std::byte>{entry->previous.data(), entry->previous_size};
        const bool changed = !std::ranges::equal(previous, snapshot.bytes);
        const auto previous_previous = std::span<const std::byte>{entry->previous_previous.data(),
                                                                  entry->previous_previous_size};
        const bool exact_aba =
            changed && entry->has_previous_previous &&
            entry->previous_previous_sequence <= std::numeric_limits<u64>::max() - 2 &&
            entry->previous_previous_sequence + 2 == snapshot.sequence &&
            std::ranges::equal(previous_previous, snapshot.bytes);

        entry->previous_previous_size = entry->previous_size;
        std::copy_n(entry->previous.begin(), entry->previous_size,
                    entry->previous_previous.begin());
        entry->previous_previous_sequence = entry->last_sequence;
        entry->has_previous_previous = true;
        entry->previous_size = static_cast<u32>(snapshot.bytes.size());
        std::ranges::copy(snapshot.bytes, entry->previous.begin());
        entry->last_sequence = snapshot.sequence;
        return {.changed = changed, .exact_aba_return = exact_aba};
    }

    void Reset() noexcept {
        for (u32 i = 0; i < count; ++i) {
            entries[i] = {};
        }
        count = 0;
    }

    [[nodiscard]] u32 EntryCount() const noexcept {
        return count;
    }

    [[nodiscard]] u64 RetainedBytes() const noexcept {
        u64 bytes{};
        for (u32 i = 0; i < count; ++i) {
            bytes += entries[i].previous_size + entries[i].previous_previous_size;
        }
        return bytes;
    }

private:
    struct Entry {
        u32 stable_identity{};
        InputAssemblyBufferToken source{};
        u64 write_serial{};
        u64 previous_previous_sequence{};
        u64 last_sequence{};
        InputAssemblyAuthority authority{InputAssemblyAuthority::Unknown};
        std::array<std::byte, MaxSnapshotBytes> previous_previous{};
        std::array<std::byte, MaxSnapshotBytes> previous{};
        u32 previous_previous_size{};
        u32 previous_size{};
        bool observed{};
        bool has_baseline{};
        bool has_previous_previous{};
    };

    static void ClearBaseline(Entry& entry) noexcept {
        entry.previous_previous_size = 0;
        entry.previous_size = 0;
        entry.has_baseline = false;
        entry.has_previous_previous = false;
    }

    static void SetBaseline(Entry& entry, const InputAssemblySnapshot& snapshot) noexcept {
        entry.source = snapshot.source;
        entry.write_serial = snapshot.write_serial;
        entry.authority = snapshot.authority;
        entry.previous_size = static_cast<u32>(snapshot.bytes.size());
        std::ranges::copy(snapshot.bytes, entry.previous.begin());
        entry.previous_previous_size = 0;
        entry.last_sequence = snapshot.sequence;
        entry.observed = true;
        entry.has_baseline = true;
        entry.has_previous_previous = false;
    }

    std::unique_ptr<Entry[]> entries;
    u32 count{};
};

template <typename Callback>
[[nodiscard]] bool CollectInputAssemblyDeviceIntegrityIfEnabled(bool enabled, Callback&& callback) {
    if (!enabled) {
        return false;
    }
    return std::forward<Callback>(callback)();
}

} // namespace AmdGpu
