// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>

#include <gtest/gtest.h>

#include "core/libraries/kernel/equeue.h"
#include "core/libraries/kernel/equeue_delivery_diagnostic.h"

namespace Diagnostic = Libraries::Kernel::EqueueDeliveryDiagnostic;
using namespace std::chrono_literals;

TEST(EqueueDeliveryDiagnostic, KeepsQueueDeliveryGenerationsSeparateAndOrdered) {
    Diagnostic::Ledger<2> ledger;
    const auto first = ledger.RegisterQueue();
    const auto second = ledger.RegisterQueue();

    ledger.RecordWaitMode(first, true);
    ledger.RecordWaitMode(first, false);
    const auto first_occurrence = ledger.RecordAccepted(first);
    const auto second_occurrence = ledger.RecordAccepted(first);
    const Diagnostic::DeliveryToken delivered{first, second_occurrence};
    ledger.RecordDequeued(delivered);
    ledger.RecordReturned(delivered);
    EXPECT_EQ(ledger.RecordAccepted(second), 1u);

    const auto snapshot = ledger.Read();
    ASSERT_EQ(snapshot.queue_count, 2u);
    EXPECT_EQ(snapshot.queues[0].zero_poll_calls, 1u);
    EXPECT_EQ(snapshot.queues[0].blocking_wait_calls, 1u);
    EXPECT_EQ(snapshot.queues[0].accepted, 2u);
    EXPECT_EQ(snapshot.queues[0].accepted_occurrence, second_occurrence);
    EXPECT_EQ(snapshot.queues[0].dequeued, 1u);
    EXPECT_EQ(snapshot.queues[0].dequeued_occurrence, second_occurrence);
    EXPECT_EQ(snapshot.queues[0].returned, 1u);
    EXPECT_EQ(snapshot.queues[0].returned_occurrence, second_occurrence);
    EXPECT_NE(first_occurrence, second_occurrence);
    EXPECT_LT(snapshot.queues[0].accepted_sequence, snapshot.queues[0].dequeued_sequence);
    EXPECT_LT(snapshot.queues[0].dequeued_sequence, snapshot.queues[0].returned_sequence);
    EXPECT_EQ(snapshot.queues[1].accepted, 1u);
    EXPECT_EQ(snapshot.queues[1].dequeued, 0u);
    EXPECT_EQ(snapshot.queues[1].returned, 0u);
}

TEST(EqueueDeliveryDiagnostic, BoundsQueueHistoryAndIgnoresInvalidSlots) {
    Diagnostic::Ledger<1> ledger;
    const auto retained = ledger.RegisterQueue();
    const auto overflow = ledger.RegisterQueue();

    EXPECT_NE(retained, Diagnostic::InvalidSlot);
    EXPECT_EQ(overflow, Diagnostic::OverflowSlot);
    EXPECT_EQ(ledger.RecordAccepted(overflow), 0u);
    ledger.RecordDequeued({overflow, 1});
    ledger.RecordReturned({overflow, 1});

    const auto snapshot = ledger.Read();
    EXPECT_EQ(snapshot.queue_count, 1u);
    EXPECT_EQ(snapshot.registration_overflow, 1u);
    EXPECT_EQ(snapshot.queues[0].accepted, 0u);
    EXPECT_EQ(snapshot.queues[0].dequeued, 0u);
    EXPECT_EQ(snapshot.queues[0].returned, 0u);
}

TEST(EqueueDeliveryDiagnostic, FrozenSnapshotRejectsLaterRecords) {
    Diagnostic::Ledger<1> ledger;
    const auto slot = ledger.RegisterQueue();
    const auto occurrence = ledger.RecordAccepted(slot);
    ledger.RecordDequeued({slot, occurrence});

    const auto snapshot = ledger.FreezeAndRead();
    ASSERT_TRUE(snapshot.frozen_stable);
    EXPECT_EQ(snapshot.queues[0].accepted, 1u);
    EXPECT_EQ(snapshot.queues[0].dequeued, 1u);
    EXPECT_EQ(snapshot.queues[0].returned, 0u);

    ledger.RecordReturned({slot, occurrence});
    const auto after = ledger.Read();
    EXPECT_EQ(after.queues[0].returned, 0u);
}

TEST(EqueueDeliveryDiagnostic, ZeroTimeoutPollSerializesWithEventMutation) {
    std::mutex event_mutex;
    std::promise<void> poll_started;
    std::atomic drained = false;
    std::unique_lock event_mutation{event_mutex};

    auto poll = std::async(std::launch::async, [&] {
        poll_started.set_value();
        return Libraries::Kernel::EqueueDetail::PollReadyEvents(event_mutex, [&] {
            drained.store(true, std::memory_order_release);
            return 1;
        });
    });

    poll_started.get_future().wait();
    EXPECT_EQ(poll.wait_for(50ms), std::future_status::timeout);
    EXPECT_FALSE(drained.load(std::memory_order_acquire));

    event_mutation.unlock();
    EXPECT_EQ(poll.get(), 1);
    EXPECT_TRUE(drained.load(std::memory_order_acquire));
}
