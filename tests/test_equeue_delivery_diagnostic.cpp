// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/kernel/equeue_delivery_diagnostic.h"

namespace Diagnostic = Libraries::Kernel::EqueueDeliveryDiagnostic;

TEST(EqueueDeliveryDiagnostic, KeepsQueueDeliveryGenerationsSeparateAndOrdered) {
    Diagnostic::Ledger<2> ledger;
    const auto first = ledger.RegisterQueue();
    const auto second = ledger.RegisterQueue();

    ledger.RecordAccepted(first);
    ledger.RecordAccepted(first);
    ledger.RecordDequeued(first);
    ledger.RecordReturned(first);
    ledger.RecordAccepted(second);

    const auto snapshot = ledger.Read();
    ASSERT_EQ(snapshot.queue_count, 2u);
    EXPECT_EQ(snapshot.queues[0].accepted, 2u);
    EXPECT_EQ(snapshot.queues[0].dequeued, 1u);
    EXPECT_EQ(snapshot.queues[0].returned, 1u);
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
    EXPECT_EQ(overflow, Diagnostic::InvalidSlot);
    ledger.RecordAccepted(overflow);
    ledger.RecordDequeued(overflow);
    ledger.RecordReturned(overflow);

    const auto snapshot = ledger.Read();
    EXPECT_EQ(snapshot.queue_count, 1u);
    EXPECT_EQ(snapshot.registration_overflow, 1u);
    EXPECT_EQ(snapshot.queues[0].accepted, 0u);
    EXPECT_EQ(snapshot.queues[0].dequeued, 0u);
    EXPECT_EQ(snapshot.queues[0].returned, 0u);
}
