// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdint>
#include <list>
#include <vector>

#include <gtest/gtest.h>

#include "core/libraries/kernel/threads/event_flag_policy.h"

namespace Libraries::Kernel {
namespace {

struct TestWaiter {
    std::uint32_t priority;
    int id;
    std::uint64_t bits{1};
    EventFlagWaitMode wait_mode{EventFlagWaitMode::And};
    EventFlagClearMode clear_mode{EventFlagClearMode::None};
    std::uint64_t result_bits{};
};

TEST(EventFlagPolicy, FifoWaitersRemainInArrivalOrder) {
    std::list<TestWaiter*> waiters;
    TestWaiter first{.priority = 700, .id = 1};
    TestWaiter second{.priority = 256, .id = 2};

    InsertEventFlagWaiter(waiters, &first, EventFlagQueueMode::Fifo);
    InsertEventFlagWaiter(waiters, &second, EventFlagQueueMode::Fifo);

    EXPECT_EQ(waiters, (std::list<TestWaiter*>{&first, &second}));
}

TEST(EventFlagPolicy, PriorityWaitersSortLowerGuestValuesFirst) {
    std::list<TestWaiter*> waiters;
    TestWaiter low{.priority = 700, .id = 1};
    TestWaiter high{.priority = 256, .id = 2};
    TestWaiter middle{.priority = 500, .id = 3};

    InsertEventFlagWaiter(waiters, &low, EventFlagQueueMode::ThreadPriority);
    InsertEventFlagWaiter(waiters, &high, EventFlagQueueMode::ThreadPriority);
    InsertEventFlagWaiter(waiters, &middle, EventFlagQueueMode::ThreadPriority);

    EXPECT_EQ(waiters, (std::list<TestWaiter*>{&high, &middle, &low}));
}

TEST(EventFlagPolicy, EqualPriorityWaitersRemainInArrivalOrder) {
    std::list<TestWaiter*> waiters;
    TestWaiter first{.priority = 500, .id = 1};
    TestWaiter second{.priority = 500, .id = 2};

    InsertEventFlagWaiter(waiters, &first, EventFlagQueueMode::ThreadPriority);
    InsertEventFlagWaiter(waiters, &second, EventFlagQueueMode::ThreadPriority);

    EXPECT_EQ(waiters, (std::list<TestWaiter*>{&first, &second}));
}

TEST(EventFlagPolicy, WaitModesMatchRequestedBits) {
    EXPECT_TRUE(IsEventFlagWaitSatisfied(0b0110, 0b0010, EventFlagWaitMode::And));
    EXPECT_FALSE(IsEventFlagWaitSatisfied(0b0010, 0b0110, EventFlagWaitMode::And));
    EXPECT_TRUE(IsEventFlagWaitSatisfied(0b0010, 0b0110, EventFlagWaitMode::Or));
    EXPECT_FALSE(IsEventFlagWaitSatisfied(0b1000, 0b0110, EventFlagWaitMode::Or));
}

TEST(EventFlagPolicy, ClearModesUpdateOnlyTheRequestedState) {
    EXPECT_EQ(ApplyEventFlagClear(0b1110, 0b0110, EventFlagClearMode::None), 0b1110u);
    EXPECT_EQ(ApplyEventFlagClear(0b1110, 0b0110, EventFlagClearMode::All), 0u);
    EXPECT_EQ(ApplyEventFlagClear(0b1110, 0b0110, EventFlagClearMode::Bits), 0b1000u);
}

TEST(EventFlagPolicy, WakesEligibleWaitersInQueueOrder) {
    std::list<TestWaiter*> waiters;
    TestWaiter low{.priority = 700, .id = 1};
    TestWaiter high{.priority = 256, .id = 2};
    InsertEventFlagWaiter(waiters, &low, EventFlagQueueMode::ThreadPriority);
    InsertEventFlagWaiter(waiters, &high, EventFlagQueueMode::ThreadPriority);
    std::uint64_t current_bits = 1;
    std::vector<int> wake_order;

    WakeEligibleEventFlagWaiters(waiters, current_bits,
                                 [&](TestWaiter* waiter) { wake_order.push_back(waiter->id); });

    EXPECT_EQ(wake_order, (std::vector<int>{2, 1}));
    EXPECT_TRUE(waiters.empty());
    EXPECT_EQ(high.result_bits, 1u);
    EXPECT_EQ(low.result_bits, 1u);
}

TEST(EventFlagPolicy, EarlierClearCanLeaveLaterWaiterBlocked) {
    std::list<TestWaiter*> waiters;
    TestWaiter low{.priority = 700, .id = 1};
    TestWaiter high{
        .priority = 256,
        .id = 2,
        .clear_mode = EventFlagClearMode::Bits,
    };
    InsertEventFlagWaiter(waiters, &low, EventFlagQueueMode::ThreadPriority);
    InsertEventFlagWaiter(waiters, &high, EventFlagQueueMode::ThreadPriority);
    std::uint64_t current_bits = 1;
    std::vector<int> wake_order;

    WakeEligibleEventFlagWaiters(waiters, current_bits,
                                 [&](TestWaiter* waiter) { wake_order.push_back(waiter->id); });

    EXPECT_EQ(wake_order, (std::vector<int>{2}));
    EXPECT_EQ(waiters, (std::list<TestWaiter*>{&low}));
    EXPECT_EQ(current_bits, 0u);
}

} // namespace
} // namespace Libraries::Kernel
