// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <future>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "core/libraries/kernel/equeue.h"
#include "core/libraries/kernel/equeue_poll.h"

namespace Libraries::Kernel {
namespace {
using namespace std::chrono_literals;

TEST(EqueuePoll, CallsCollectorOnceAndReturnsEmptyWithoutWaitingForAnEvent) {
    std::mutex mutex;
    int calls = 0;
    EXPECT_EQ(PollEventQueue(mutex,
                             [&] {
                                 ++calls;
                                 return 0;
                             }),
              0);
    EXPECT_EQ(calls, 1);
}

TEST(EqueuePoll, RetainsActualMutexFromEventCopyThroughClear) {
    std::mutex mutex;
    EqueueEvent event{};
    event.event.flags = OrbisKernelEvent::Flags::Clear;
    event.Trigger(reinterpret_cast<void*>(42));
    OrbisKernelEvent copied{};
    std::promise<void> copied_signal;
    std::promise<void> finish_clear;
    auto finish = finish_clear.get_future();
    auto poll = std::async(std::launch::async, [&] {
        return PollEventQueue(mutex, [&] {
            copied = event.event;
            copied_signal.set_value();
            finish.wait();
            event.Clear();
            return 1;
        });
    });
    const auto copied_status = copied_signal.get_future().wait_for(1s);
    // The collector is latched between copy and clear. Probe from a DIFFERENT thread;
    // try_lock on one's own non-recursive mutex would not be a valid ownership test.
    const bool unprotected = mutex.try_lock();
    if (unprotected) {
        mutex.unlock();
    }
    finish_clear.set_value(); // Rescue before any assertion or future destruction.
    const auto count = poll.get();
    EXPECT_EQ(copied_status, std::future_status::ready);
    EXPECT_FALSE(unprotected);
    EXPECT_EQ(count, 1);
    EXPECT_EQ(copied.data, 42u);
    EXPECT_FALSE(event.IsTriggered());
}

TEST(EqueuePoll, RetainsActualMutexThroughOneShotRemoval) {
    std::mutex mutex;
    std::vector<EqueueEvent> events(1);
    events.front().event.flags = OrbisKernelEvent::Flags::OneShot;
    events.front().Trigger(reinterpret_cast<void*>(17));
    std::promise<void> removing_signal;
    std::promise<void> finish_remove;
    auto finish = finish_remove.get_future();
    auto poll = std::async(std::launch::async, [&] {
        return PollEventQueue(mutex, [&] {
            const auto flags = events.front().event.flags;
            removing_signal.set_value();
            finish.wait();
            if (flags & OrbisKernelEvent::Flags::OneShot) {
                events.erase(events.begin());
            }
            return 1;
        });
    });
    const auto removing_status = removing_signal.get_future().wait_for(1s);
    const bool unprotected = mutex.try_lock();
    if (unprotected) {
        mutex.unlock();
    }
    finish_remove.set_value();
    const auto count = poll.get();
    EXPECT_EQ(removing_status, std::future_status::ready);
    EXPECT_FALSE(unprotected);
    EXPECT_EQ(count, 1);
    EXPECT_TRUE(events.empty());
}

TEST(EqueuePoll, WaitsForCurrentWriterButNotForANewEvent) {
    std::mutex mutex;
    std::unique_lock writer{mutex};
    std::promise<void> started_signal;
    auto poll = std::async(std::launch::async, [&] {
        started_signal.set_value();
        return PollEventQueue(mutex, [] { return 0; });
    });
    started_signal.get_future().wait();
    const auto before_unlock = poll.wait_for(20ms);
    writer.unlock(); // Always release before assertions/join.
    const auto count = poll.get();
    EXPECT_EQ(before_unlock, std::future_status::timeout);
    EXPECT_EQ(count, 0);
}

TEST(EqueuePoll, ClearThenRetriggerRemainsObservableOnTheNextPoll) {
    std::mutex mutex;
    EqueueEvent event{};
    event.event.flags = OrbisKernelEvent::Flags::Clear;
    event.Trigger(reinterpret_cast<void*>(42));
    auto collect = [&] {
        if (!event.IsTriggered()) {
            return u64{0};
        }
        const auto data = event.event.data;
        event.Clear();
        return data;
    };
    EXPECT_EQ(PollEventQueue(mutex, collect), 42u);
    EXPECT_EQ(PollEventQueue(mutex, collect), 0u);
    {
        std::scoped_lock writer{mutex};
        event.Trigger(reinterpret_cast<void*>(84));
    }
    EXPECT_EQ(PollEventQueue(mutex, collect), 84u);
    EXPECT_EQ(PollEventQueue(mutex, collect), 0u);
}

TEST(EqueuePoll, CollectorExceptionReleasesTheMutex) {
    std::mutex mutex;
    EXPECT_THROW(PollEventQueue(mutex, []() -> int { throw std::runtime_error{"collector"}; }),
                 std::runtime_error);
    const bool released = mutex.try_lock();
    if (released) {
        mutex.unlock();
    }
    EXPECT_TRUE(released);
}

TEST(EqueuePollWiring, ZeroTimeoutCollectsUnderLockWithoutRecursivelyLockingBlockingWaits) {
    std::ifstream input{SHADPS4_EQUEUE_SOURCE_PATH};
    ASSERT_TRUE(input);
    std::string source{std::istreambuf_iterator<char>{input}, {}};
    std::erase_if(source, [](unsigned char value) { return std::isspace(value); });
    const auto start = source.find("intEqueueInternal::WaitForEvents(");
    const auto end = source.find("boolEqueueInternal::TriggerEvent(", start);
    ASSERT_NE(start, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto wait = source.substr(start, end - start);
    const auto poll_end = wait.find("constautomicros=");
    ASSERT_NE(poll_end, std::string::npos);
    const auto poll = wait.substr(0, poll_end);
    EXPECT_NE(poll.find("if(timo!=nullptr&&*timo==0)"), std::string::npos);
    EXPECT_NE(poll.find("returnPollEventQueue(m_mutex,[&]{returnGetTriggeredEvents(ev,num);});"),
              std::string::npos);
    const auto blocking = wait.substr(poll_end);
    EXPECT_EQ(blocking.find("PollEventQueue("), std::string::npos);
    EXPECT_NE(
        blocking.find("constautopredicate=[&]{count=GetTriggeredEvents(ev,num);returncount>0;}"),
        std::string::npos);
    EXPECT_NE(blocking.find("std::unique_locklock{m_mutex};m_cond.wait(lock,predicate);"),
              std::string::npos);
    EXPECT_NE(blocking.find("std::unique_locklock{m_mutex};m_cond.wait_for(lock,"),
              std::string::npos);
    const auto collect_start = source.find("intEqueueInternal::GetTriggeredEvents(");
    const auto collect_end = source.find("boolEqueueInternal::AddSmallTimer(", collect_start);
    ASSERT_NE(collect_start, std::string::npos);
    ASSERT_NE(collect_end, std::string::npos);
    const auto collect = source.substr(collect_start, collect_end - collect_start);
    EXPECT_EQ(collect.find("m_mutex"), std::string::npos);
    EXPECT_EQ(collect.find("PollEventQueue("), std::string::npos);
    EXPECT_NE(collect.find("ev[count++]=it->event;"), std::string::npos);
    EXPECT_NE(collect.find("it->Clear();"), std::string::npos);
    EXPECT_NE(collect.find("it=m_events.erase(it);"), std::string::npos);
}

} // namespace
} // namespace Libraries::Kernel
