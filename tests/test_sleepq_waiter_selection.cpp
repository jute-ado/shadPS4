// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdint>

#include <gtest/gtest.h>

#include "core/libraries/kernel/threads/sleepq.h"

namespace Libraries::Kernel {
namespace {

Pthread* FakeThread(std::uintptr_t value) {
    return reinterpret_cast<Pthread*>(value);
}

TEST(SleepqWaiterSelection, UntargetedWakeSelectsAQueuedWaiter) {
    SleepQueue queue{};
    Pthread* first = FakeThread(1);
    Pthread* second = FakeThread(2);
    queue.sq_blocked.push_front(first);
    queue.sq_blocked.push_front(second);

    EXPECT_EQ(SelectSleepqWaiter(&queue, nullptr), second);
}

TEST(SleepqWaiterSelection, TargetedWakeSelectsTheRequestedQueuedWaiter) {
    SleepQueue queue{};
    Pthread* first = FakeThread(1);
    Pthread* second = FakeThread(2);
    queue.sq_blocked.push_front(first);
    queue.sq_blocked.push_front(second);

    EXPECT_EQ(SelectSleepqWaiter(&queue, first), first);
}

TEST(SleepqWaiterSelection, TargetedWakeRejectsAThreadWaitingElsewhere) {
    SleepQueue queue{};
    queue.sq_blocked.push_front(FakeThread(1));

    EXPECT_EQ(SelectSleepqWaiter(&queue, FakeThread(2)), nullptr);
}

TEST(SleepqWaiterSelection, EmptyQueueHasNoWaiterToWake) {
    SleepQueue queue{};

    EXPECT_EQ(SelectSleepqWaiter(&queue, nullptr), nullptr);
}

} // namespace
} // namespace Libraries::Kernel
