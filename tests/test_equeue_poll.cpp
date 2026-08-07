// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>

#include <gtest/gtest.h>

#include "core/libraries/kernel/equeue.h"

using namespace std::chrono_literals;

TEST(EqueuePoll, ZeroTimeoutPollSerializesWithTriggerMutation) {
    std::mutex event_mutex;
    std::promise<void> poll_started;
    std::atomic drained = false;
    std::unique_lock trigger_mutation{event_mutex};

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

    trigger_mutation.unlock();
    EXPECT_EQ(poll.get(), 1);
    EXPECT_TRUE(drained.load(std::memory_order_acquire));
}
