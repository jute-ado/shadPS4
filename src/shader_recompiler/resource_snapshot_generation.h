// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <new>
#include <optional>
#include <ranges>
#include <type_traits>

#include "common/types.h"

namespace Shader {

static constexpr u32 MaxResourceSnapshotValidationCaptures = 4;
static constexpr size_t MaxResourceSnapshotDiagnosticWords = 4096;
static constexpr u32 NumResourceSnapshotLogicalStages = 6;
static constexpr u64 MaxResourceSnapshotMismatchReports = 64;

enum class ResourceSnapshotGenerationStatus : u8 {
    Disabled,
    Stable,
    ChangedThenStable,
    CaptureUnavailable,
    RetryExhausted,
    CapacityExceeded,
};

struct ResourceSnapshotGenerationObservation {
    ResourceSnapshotGenerationStatus status{ResourceSnapshotGenerationStatus::Disabled};
    u32 validation_captures{};
    bool available{};
    bool rendered_generation_stable{};
    bool user_data_changed{};
    bool resource_data_changed{};
};

/**
 * Checks whether the complete snapshot retained for rendering is followed by the same generation.
 * Later captures are diagnostic-only and never replace or modify the retained snapshot.
 */
template <typename Snapshot, typename Capture>
    requires std::ranges::contiguous_range<Snapshot> && std::ranges::sized_range<Snapshot>
ResourceSnapshotGenerationObservation ObserveResourceSnapshotGeneration(
    const Snapshot& rendered_snapshot, size_t user_data_words, bool enabled,
    u32 max_validation_captures, size_t max_snapshot_words, Capture&& capture) {
    ResourceSnapshotGenerationObservation observation{};
    if (!enabled) {
        return observation;
    }

    const size_t snapshot_words = std::ranges::size(rendered_snapshot);
    if (max_validation_captures == 0 ||
        max_validation_captures > MaxResourceSnapshotValidationCaptures ||
        snapshot_words > max_snapshot_words || user_data_words > snapshot_words) {
        observation.status = ResourceSnapshotGenerationStatus::CapacityExceeded;
        return observation;
    }

    using SnapshotType = std::remove_cvref_t<Snapshot>;
    const auto classify_difference = [&](const SnapshotType& before, const SnapshotType& after) {
        if (std::ranges::size(before) != snapshot_words ||
            std::ranges::size(after) != snapshot_words) {
            observation.resource_data_changed = true;
            return;
        }

        const auto before_begin = std::ranges::begin(before);
        const auto after_begin = std::ranges::begin(after);
        observation.user_data_changed |=
            !std::ranges::equal(before_begin, before_begin + user_data_words, after_begin,
                                after_begin + user_data_words);
        observation.resource_data_changed |=
            !std::ranges::equal(before_begin + user_data_words, std::ranges::end(before),
                                after_begin + user_data_words, std::ranges::end(after));
    };

    const SnapshotType* previous = &rendered_snapshot;
    std::optional<SnapshotType> previous_capture;
    bool changed = false;
    for (u32 capture_index = 0; capture_index < max_validation_captures; ++capture_index) {
        auto next = std::invoke(capture);
        ++observation.validation_captures;
        if (!next.has_value()) {
            observation.status = ResourceSnapshotGenerationStatus::CaptureUnavailable;
            return observation;
        }

        if (std::ranges::equal(*previous, *next)) {
            observation.available = true;
            observation.rendered_generation_stable = !changed;
            observation.status = changed ? ResourceSnapshotGenerationStatus::ChangedThenStable
                                         : ResourceSnapshotGenerationStatus::Stable;
            return observation;
        }

        classify_difference(*previous, *next);
        changed = true;
        previous_capture = std::move(*next);
        previous = &*previous_capture;
    }

    observation.status = ResourceSnapshotGenerationStatus::RetryExhausted;
    return observation;
}

struct ResourceSnapshotGenerationDiagnosticSnapshot {
    bool stable{};
    u32 writers_before{};
    u32 writers_after{};
    u64 sequence_before{};
    u64 sequence_after{};
    u64 frames{};
    u64 draws{};
    u64 dispatches{};
    u64 observations{};
    u64 stable_generations{};
    u64 changed_then_stable{};
    u64 capture_unavailable{};
    u64 retry_exhausted{};
    u64 capacity_exceeded{};
    u64 validation_captures{};
    u64 user_data_changes{};
    u64 resource_data_changes{};
    u64 observed_words{};
    u64 maximum_snapshot_words{};
    std::array<u64, NumResourceSnapshotLogicalStages> stage_observations{};
    u64 abnormal_occurrences{};
    u64 last_frame{};
    u64 last_draw{};
    u64 last_dispatch{};
    u32 last_stage{};
    u32 last_snapshot_words{};
    ResourceSnapshotGenerationStatus last_status{ResourceSnapshotGenerationStatus::Disabled};
};

struct ResourceSnapshotGenerationEvent {
    bool should_report{};
    u64 occurrence{};
    u64 frame{};
    u64 draw{};
    u64 dispatch{};
};

class ResourceSnapshotGenerationDiagnostic {
public:
    void ObserveFrameBoundary() noexcept {
        BeginUpdate();
        frames.fetch_add(1, std::memory_order_relaxed);
        EndUpdate();
    }

    void ObserveDraw() noexcept {
        BeginUpdate();
        draws.fetch_add(1, std::memory_order_relaxed);
        EndUpdate();
    }

    void ObserveDispatch() noexcept {
        BeginUpdate();
        dispatches.fetch_add(1, std::memory_order_relaxed);
        EndUpdate();
    }

    ResourceSnapshotGenerationEvent ObserveStage(
        u32 stage, size_t snapshot_words,
        const ResourceSnapshotGenerationObservation& observation) noexcept {
        BeginUpdate();
        observations.fetch_add(1, std::memory_order_relaxed);
        validation_captures.fetch_add(observation.validation_captures, std::memory_order_relaxed);
        observed_words.fetch_add(snapshot_words, std::memory_order_relaxed);
        UpdateMaximum(maximum_snapshot_words, snapshot_words);
        if (stage < stage_observations.size()) {
            stage_observations[stage].fetch_add(1, std::memory_order_relaxed);
        }
        user_data_changes.fetch_add(observation.user_data_changed, std::memory_order_relaxed);
        resource_data_changes.fetch_add(observation.resource_data_changed,
                                        std::memory_order_relaxed);

        switch (observation.status) {
        case ResourceSnapshotGenerationStatus::Stable:
            stable_generations.fetch_add(1, std::memory_order_relaxed);
            EndUpdate();
            return {};
        case ResourceSnapshotGenerationStatus::ChangedThenStable:
            changed_then_stable.fetch_add(1, std::memory_order_relaxed);
            break;
        case ResourceSnapshotGenerationStatus::CaptureUnavailable:
            capture_unavailable.fetch_add(1, std::memory_order_relaxed);
            break;
        case ResourceSnapshotGenerationStatus::RetryExhausted:
            retry_exhausted.fetch_add(1, std::memory_order_relaxed);
            break;
        case ResourceSnapshotGenerationStatus::CapacityExceeded:
            capacity_exceeded.fetch_add(1, std::memory_order_relaxed);
            break;
        case ResourceSnapshotGenerationStatus::Disabled:
            EndUpdate();
            return {};
        }

        const u64 occurrence = abnormal_occurrences.fetch_add(1, std::memory_order_relaxed) + 1;
        const u64 frame = frames.load(std::memory_order_relaxed);
        const u64 draw = draws.load(std::memory_order_relaxed);
        const u64 dispatch = dispatches.load(std::memory_order_relaxed);
        last_frame.store(frame, std::memory_order_relaxed);
        last_draw.store(draw, std::memory_order_relaxed);
        last_dispatch.store(dispatch, std::memory_order_relaxed);
        last_stage.store(stage, std::memory_order_relaxed);
        last_snapshot_words.store(static_cast<u32>(snapshot_words), std::memory_order_relaxed);
        last_status.store(observation.status, std::memory_order_relaxed);
        EndUpdate();
        return {
            .should_report = occurrence <= MaxResourceSnapshotMismatchReports,
            .occurrence = occurrence,
            .frame = frame,
            .draw = draw,
            .dispatch = dispatch,
        };
    }

    [[nodiscard]] ResourceSnapshotGenerationDiagnosticSnapshot Read() const noexcept {
        ResourceSnapshotGenerationDiagnosticSnapshot snapshot{};
        snapshot.writers_before = active_writers.load(std::memory_order_acquire);
        snapshot.sequence_before = sequence.load(std::memory_order_acquire);
        snapshot.frames = frames.load(std::memory_order_relaxed);
        snapshot.draws = draws.load(std::memory_order_relaxed);
        snapshot.dispatches = dispatches.load(std::memory_order_relaxed);
        snapshot.observations = observations.load(std::memory_order_relaxed);
        snapshot.stable_generations = stable_generations.load(std::memory_order_relaxed);
        snapshot.changed_then_stable = changed_then_stable.load(std::memory_order_relaxed);
        snapshot.capture_unavailable = capture_unavailable.load(std::memory_order_relaxed);
        snapshot.retry_exhausted = retry_exhausted.load(std::memory_order_relaxed);
        snapshot.capacity_exceeded = capacity_exceeded.load(std::memory_order_relaxed);
        snapshot.validation_captures = validation_captures.load(std::memory_order_relaxed);
        snapshot.user_data_changes = user_data_changes.load(std::memory_order_relaxed);
        snapshot.resource_data_changes = resource_data_changes.load(std::memory_order_relaxed);
        snapshot.observed_words = observed_words.load(std::memory_order_relaxed);
        snapshot.maximum_snapshot_words = maximum_snapshot_words.load(std::memory_order_relaxed);
        for (u32 i = 0; i < stage_observations.size(); ++i) {
            snapshot.stage_observations[i] = stage_observations[i].load(std::memory_order_relaxed);
        }
        snapshot.abnormal_occurrences = abnormal_occurrences.load(std::memory_order_relaxed);
        snapshot.last_frame = last_frame.load(std::memory_order_relaxed);
        snapshot.last_draw = last_draw.load(std::memory_order_relaxed);
        snapshot.last_dispatch = last_dispatch.load(std::memory_order_relaxed);
        snapshot.last_stage = last_stage.load(std::memory_order_relaxed);
        snapshot.last_snapshot_words = last_snapshot_words.load(std::memory_order_relaxed);
        snapshot.last_status = last_status.load(std::memory_order_relaxed);
        snapshot.sequence_after = sequence.load(std::memory_order_acquire);
        snapshot.writers_after = active_writers.load(std::memory_order_acquire);
        snapshot.stable = snapshot.writers_before == 0 && snapshot.writers_after == 0 &&
                          snapshot.sequence_before == snapshot.sequence_after;
        return snapshot;
    }

    bool TryMarkReported() noexcept {
        return !reported.exchange(true, std::memory_order_acq_rel);
    }

private:
    static void UpdateMaximum(std::atomic<u64>& target, u64 value) noexcept {
        u64 current = target.load(std::memory_order_relaxed);
        while (current < value &&
               !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
        }
    }

    void BeginUpdate() noexcept {
        active_writers.fetch_add(1, std::memory_order_acq_rel);
    }

    void EndUpdate() noexcept {
        sequence.fetch_add(1, std::memory_order_release);
        active_writers.fetch_sub(1, std::memory_order_release);
    }

    std::atomic<u32> active_writers{};
    std::atomic<u64> sequence{};
    std::atomic<u64> frames{};
    std::atomic<u64> draws{};
    std::atomic<u64> dispatches{};
    std::atomic<u64> observations{};
    std::atomic<u64> stable_generations{};
    std::atomic<u64> changed_then_stable{};
    std::atomic<u64> capture_unavailable{};
    std::atomic<u64> retry_exhausted{};
    std::atomic<u64> capacity_exceeded{};
    std::atomic<u64> validation_captures{};
    std::atomic<u64> user_data_changes{};
    std::atomic<u64> resource_data_changes{};
    std::atomic<u64> observed_words{};
    std::atomic<u64> maximum_snapshot_words{};
    std::array<std::atomic<u64>, NumResourceSnapshotLogicalStages> stage_observations{};
    std::atomic<u64> abnormal_occurrences{};
    std::atomic<u64> last_frame{};
    std::atomic<u64> last_draw{};
    std::atomic<u64> last_dispatch{};
    std::atomic<u32> last_stage{};
    std::atomic<u32> last_snapshot_words{};
    std::atomic<ResourceSnapshotGenerationStatus> last_status{};
    std::atomic<bool> reported{};
};

inline bool ResourceSnapshotGenerationDiagnosticEnabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_DIAGNOSTIC_SHADER_RESOURCE_GENERATION");
        return value != nullptr &&
               (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
                std::strcmp(value, "on") == 0);
    }();
    return enabled;
}

inline ResourceSnapshotGenerationDiagnostic& GetResourceSnapshotGenerationDiagnostic() noexcept {
    static ResourceSnapshotGenerationDiagnostic diagnostic;
    return diagnostic;
}

inline void ObserveResourceSnapshotFrameBoundary() noexcept {
    if (ResourceSnapshotGenerationDiagnosticEnabled()) {
        GetResourceSnapshotGenerationDiagnostic().ObserveFrameBoundary();
    }
}

inline void ObserveResourceSnapshotDraw() noexcept {
    if (ResourceSnapshotGenerationDiagnosticEnabled()) {
        GetResourceSnapshotGenerationDiagnostic().ObserveDraw();
    }
}

inline void ObserveResourceSnapshotDispatch() noexcept {
    if (ResourceSnapshotGenerationDiagnosticEnabled()) {
        GetResourceSnapshotGenerationDiagnostic().ObserveDispatch();
    }
}

} // namespace Shader
