// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <string_view>

#include "common/types.h"

namespace AmdGpu::EopBoundaryDiagnostic {

struct Snapshot {
    u64 sequence_before{};
    u64 submit_done_enqueued{};
    u64 submit_done_enqueued_sequence{};
    u64 submit_done_consumed{};
    u64 submit_done_consumed_sequence{};
    u64 submit_done_boundary_completed{};
    u64 submit_done_boundary_completed_sequence{};
    u64 eop_decoded{};
    u64 eop_decoded_sequence{};
    u64 eop_irq_requested{};
    u64 eop_irq_requested_sequence{};
    u64 eop_irq_delivered{};
    u64 eop_irq_delivered_sequence{};
    u64 sequence_after{};
};

struct State {
    std::atomic<u64> sequence{};
    std::atomic<u64> submit_done_enqueued{};
    std::atomic<u64> submit_done_enqueued_sequence{};
    std::atomic<u64> submit_done_consumed{};
    std::atomic<u64> submit_done_consumed_sequence{};
    std::atomic<u64> submit_done_boundary_completed{};
    std::atomic<u64> submit_done_boundary_completed_sequence{};
    std::atomic<u64> eop_decoded{};
    std::atomic<u64> eop_decoded_sequence{};
    std::atomic<u64> eop_irq_requested{};
    std::atomic<u64> eop_irq_requested_sequence{};
    std::atomic<u64> eop_irq_delivered{};
    std::atomic<u64> eop_irq_delivered_sequence{};
};

[[nodiscard]] inline bool Enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_DIAGNOSTIC_GNM_EOP_BOUNDARY");
        return value != nullptr && std::string_view{value} == "1";
    }();
    return enabled;
}

[[nodiscard]] inline State& GetState() noexcept {
    static State state{};
    return state;
}

inline void Record(std::atomic<u64>& count, std::atomic<u64>& last_sequence) noexcept {
    if (!Enabled()) {
        return;
    }
    auto& state = GetState();
    const u64 sequence = state.sequence.fetch_add(1) + 1;
    count.fetch_add(1);
    last_sequence.store(sequence);
}

inline void RecordSubmitDoneEnqueued() noexcept {
    auto& state = GetState();
    Record(state.submit_done_enqueued, state.submit_done_enqueued_sequence);
}

inline void RecordSubmitDoneConsumed() noexcept {
    auto& state = GetState();
    Record(state.submit_done_consumed, state.submit_done_consumed_sequence);
}

inline void RecordSubmitDoneBoundaryCompleted() noexcept {
    auto& state = GetState();
    Record(state.submit_done_boundary_completed, state.submit_done_boundary_completed_sequence);
}

inline void RecordEopDecoded() noexcept {
    auto& state = GetState();
    Record(state.eop_decoded, state.eop_decoded_sequence);
}

inline void RecordEopIrqDelivered() noexcept {
    auto& state = GetState();
    Record(state.eop_irq_delivered, state.eop_irq_delivered_sequence);
}

inline void RecordEopIrqRequested() noexcept {
    auto& state = GetState();
    Record(state.eop_irq_requested, state.eop_irq_requested_sequence);
}

[[nodiscard]] inline Snapshot Read() noexcept {
    const auto& state = GetState();
    Snapshot snapshot{};
    snapshot.sequence_before = state.sequence.load();
    snapshot.submit_done_enqueued = state.submit_done_enqueued.load();
    snapshot.submit_done_enqueued_sequence = state.submit_done_enqueued_sequence.load();
    snapshot.submit_done_consumed = state.submit_done_consumed.load();
    snapshot.submit_done_consumed_sequence = state.submit_done_consumed_sequence.load();
    snapshot.submit_done_boundary_completed = state.submit_done_boundary_completed.load();
    snapshot.submit_done_boundary_completed_sequence =
        state.submit_done_boundary_completed_sequence.load();
    snapshot.eop_decoded = state.eop_decoded.load();
    snapshot.eop_decoded_sequence = state.eop_decoded_sequence.load();
    snapshot.eop_irq_requested = state.eop_irq_requested.load();
    snapshot.eop_irq_requested_sequence = state.eop_irq_requested_sequence.load();
    snapshot.eop_irq_delivered = state.eop_irq_delivered.load();
    snapshot.eop_irq_delivered_sequence = state.eop_irq_delivered_sequence.load();
    snapshot.sequence_after = state.sequence.load();
    return snapshot;
}

} // namespace AmdGpu::EopBoundaryDiagnostic
