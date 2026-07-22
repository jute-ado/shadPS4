// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/kernel/threads/condvar_wait_state.h"

namespace Libraries::Kernel {
namespace {

TEST(CondvarWaitState, RemovedWaiterCompletesEvenWhenItsHostWaitTimedOut) {
    EXPECT_EQ(DecideCondWait(false, false, true), CondWaitDecision::Woken);
}

TEST(CondvarWaitState, CancellationWinsWhileWaiterRemainsQueued) {
    EXPECT_EQ(DecideCondWait(true, true, false), CondWaitDecision::Canceled);
}

TEST(CondvarWaitState, HostTimeoutRemovesAWaiterStillQueued) {
    EXPECT_EQ(DecideCondWait(true, false, true), CondWaitDecision::TimedOut);
}

TEST(CondvarWaitState, StaleWakeRetriesWhileWaiterRemainsQueued) {
    EXPECT_EQ(DecideCondWait(true, false, false), CondWaitDecision::Retry);
}

} // namespace
} // namespace Libraries::Kernel
