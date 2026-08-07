// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <thread>

#include "common/types.h"

namespace Libraries::Kernel::EqueueDeliveryDiagnostic {

constexpr u32 InvalidSlot = std::numeric_limits<u32>::max();
constexpr u32 OverflowSlot = InvalidSlot - 1;
constexpr u32 MaxQueues = 8;

struct DeliveryToken {
    u32 slot{InvalidSlot};
    u64 occurrence{};
};

struct QueueSnapshot {
    u64 zero_poll_calls{};
    u64 blocking_wait_calls{};
    u64 accepted{};
    u64 accepted_occurrence{};
    u64 accepted_sequence{};
    u64 dequeued{};
    u64 dequeued_occurrence{};
    u64 dequeued_sequence{};
    u64 returned{};
    u64 returned_occurrence{};
    u64 returned_sequence{};
};

template <u32 QueueCapacity>
struct Snapshot {
    bool frozen_stable{};
    u64 sequence{};
    u32 queue_count{};
    u64 registration_overflow{};
    std::array<QueueSnapshot, QueueCapacity> queues{};
};

template <u32 QueueCapacity>
class Ledger {
    struct QueueState {
        std::atomic<u64> zero_poll_calls{};
        std::atomic<u64> blocking_wait_calls{};
        std::atomic<u64> accepted{};
        std::atomic<u64> accepted_occurrence{};
        std::atomic<u64> accepted_sequence{};
        std::atomic<u64> dequeued{};
        std::atomic<u64> dequeued_occurrence{};
        std::atomic<u64> dequeued_sequence{};
        std::atomic<u64> returned{};
        std::atomic<u64> returned_occurrence{};
        std::atomic<u64> returned_sequence{};
    };

public:
    [[nodiscard]] u32 RegisterQueue() noexcept {
        if (!BeginRecord()) {
            return OverflowSlot;
        }
        const u32 slot = next_slot.fetch_add(1);
        if (slot >= QueueCapacity) {
            registration_overflow.fetch_add(1);
            EndRecord();
            return OverflowSlot;
        }
        EndRecord();
        return slot;
    }

    [[nodiscard]] u64 RecordAccepted(u32 slot) noexcept {
        if (slot >= QueueCapacity || !BeginRecord()) {
            return 0;
        }
        auto& queue = queues[slot];
        const u64 occurrence = queue.accepted.fetch_add(1) + 1;
        const u64 current_sequence = sequence.fetch_add(1) + 1;
        queue.accepted_occurrence.store(occurrence);
        queue.accepted_sequence.store(current_sequence);
        EndRecord();
        return occurrence;
    }

    void RecordWaitMode(u32 slot, bool zero_poll) noexcept {
        if (slot >= QueueCapacity || !BeginRecord()) {
            return;
        }
        auto& queue = queues[slot];
        (zero_poll ? queue.zero_poll_calls : queue.blocking_wait_calls).fetch_add(1);
        sequence.fetch_add(1);
        EndRecord();
    }

    void RecordDequeued(DeliveryToken token) noexcept {
        RecordStage(token, &QueueState::dequeued, &QueueState::dequeued_occurrence,
                    &QueueState::dequeued_sequence);
    }

    void RecordReturned(DeliveryToken token) noexcept {
        RecordStage(token, &QueueState::returned, &QueueState::returned_occurrence,
                    &QueueState::returned_sequence);
    }

    [[nodiscard]] Snapshot<QueueCapacity> Read() const noexcept {
        return ReadImpl(false);
    }

    [[nodiscard]] Snapshot<QueueCapacity> FreezeAndRead() noexcept {
        frozen.store(true);
        bool stable = false;
        for (u32 attempt = 0; attempt < 4096; ++attempt) {
            if (in_flight.load() == 0) {
                stable = true;
                break;
            }
            std::this_thread::yield();
        }
        return ReadImpl(stable);
    }

private:
    [[nodiscard]] bool BeginRecord() noexcept {
        if (frozen.load()) {
            return false;
        }
        in_flight.fetch_add(1);
        if (frozen.load()) {
            in_flight.fetch_sub(1);
            return false;
        }
        return true;
    }

    void EndRecord() noexcept {
        in_flight.fetch_sub(1);
    }

    void RecordStage(DeliveryToken token, std::atomic<u64> QueueState::* count,
                     std::atomic<u64> QueueState::* occurrence,
                     std::atomic<u64> QueueState::* last_sequence) noexcept {
        if (token.slot >= QueueCapacity || token.occurrence == 0 || !BeginRecord()) {
            return;
        }
        auto& queue = queues[token.slot];
        (queue.*count).fetch_add(1);
        const u64 current_sequence = sequence.fetch_add(1) + 1;
        (queue.*occurrence).store(token.occurrence);
        (queue.*last_sequence).store(current_sequence);
        EndRecord();
    }

    [[nodiscard]] Snapshot<QueueCapacity> ReadImpl(bool frozen_stable) const noexcept {
        Snapshot<QueueCapacity> snapshot{};
        snapshot.frozen_stable = frozen_stable;
        snapshot.sequence = sequence.load();
        snapshot.queue_count = std::min(next_slot.load(), QueueCapacity);
        snapshot.registration_overflow = registration_overflow.load();
        for (u32 slot = 0; slot < snapshot.queue_count; ++slot) {
            const auto& source = queues[slot];
            auto& target = snapshot.queues[slot];
            target.zero_poll_calls = source.zero_poll_calls.load();
            target.blocking_wait_calls = source.blocking_wait_calls.load();
            target.accepted = source.accepted.load();
            target.accepted_occurrence = source.accepted_occurrence.load();
            target.accepted_sequence = source.accepted_sequence.load();
            target.dequeued = source.dequeued.load();
            target.dequeued_occurrence = source.dequeued_occurrence.load();
            target.dequeued_sequence = source.dequeued_sequence.load();
            target.returned = source.returned.load();
            target.returned_occurrence = source.returned_occurrence.load();
            target.returned_sequence = source.returned_sequence.load();
        }
        return snapshot;
    }

    std::atomic<bool> frozen{};
    std::atomic<u32> in_flight{};
    std::atomic<u64> sequence{};
    std::atomic<u32> next_slot{};
    std::atomic<u64> registration_overflow{};
    std::array<QueueState, QueueCapacity> queues{};
};

[[nodiscard]] inline bool Enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_DIAGNOSTIC_GNM_EQUEUE_DELIVERY");
        return value != nullptr && std::string_view{value} == "1";
    }();
    return enabled;
}

[[nodiscard]] inline Ledger<MaxQueues>& GetLedger() noexcept {
    static Ledger<MaxQueues> ledger{};
    return ledger;
}

[[nodiscard]] inline u32 RegisterQueue() noexcept {
    return Enabled() ? GetLedger().RegisterQueue() : InvalidSlot;
}

[[nodiscard]] inline u64 RecordAccepted(u32 slot) noexcept {
    return Enabled() ? GetLedger().RecordAccepted(slot) : 0;
}

inline void RecordWaitMode(u32 slot, bool zero_poll) noexcept {
    if (Enabled()) {
        GetLedger().RecordWaitMode(slot, zero_poll);
    }
}

inline void RecordDequeued(DeliveryToken token) noexcept {
    if (Enabled()) {
        GetLedger().RecordDequeued(token);
    }
}

inline void RecordReturned(DeliveryToken token) noexcept {
    if (Enabled()) {
        GetLedger().RecordReturned(token);
    }
}

[[nodiscard]] inline Snapshot<MaxQueues> FreezeAndRead() noexcept {
    return GetLedger().FreezeAndRead();
}

} // namespace Libraries::Kernel::EqueueDeliveryDiagnostic
