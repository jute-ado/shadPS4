// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "core/libraries/gnmdriver/submission_gate.h"
#include "core/libraries/gnmdriver/submission_transaction.h"

using namespace std::chrono_literals;

namespace Libraries::GnmDriver {
namespace {

TEST(GnmSubmissionGate, PendingBoundaryBlocksTheNextSubmissionUntilGpuAcknowledgement) {
    SubmissionGate gate;
    auto complete_boundary = gate.BeginBoundary();

    auto entrant = std::async(std::launch::async, [&] {
        auto guard = gate.Enter();
        return true;
    });
    EXPECT_EQ(entrant.wait_for(20ms), std::future_status::timeout);

    complete_boundary();

    EXPECT_EQ(entrant.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(entrant.get());
}

TEST(GnmSubmissionGate, PendingBoundaryBlocksGpuVisibleMappingWork) {
    SubmissionGate gate;
    auto complete_boundary = gate.BeginBoundary();

    auto mapping = std::async(std::launch::async, [&] {
        gate.WaitForBoundary();
        return true;
    });
    EXPECT_EQ(mapping.wait_for(20ms), std::future_status::timeout);

    complete_boundary();

    EXPECT_EQ(mapping.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(mapping.get());
}

TEST(GnmSubmissionGate, SubmitDoneCannotSplitAnInProgressSubmission) {
    SubmissionGate gate;
    auto submission = gate.Enter();

    std::atomic<bool> close_started{};
    auto closer = std::async(std::launch::async, [&] {
        close_started = true;
        return gate.BeginBoundary();
    });
    while (!close_started.load()) {
        std::this_thread::yield();
    }

    EXPECT_EQ(closer.wait_for(20ms), std::future_status::timeout);
    submission = {};

    EXPECT_EQ(closer.wait_for(1s), std::future_status::ready);
    auto complete_boundary = closer.get();
    EXPECT_FALSE(gate.IsBoundaryOpen());
    complete_boundary();
}

TEST(GnmSubmissionGate, StaleCompletionCannotAcknowledgeANewerBoundary) {
    SubmissionGate gate;

    auto complete_first = gate.BeginBoundary();
    complete_first();
    auto complete_second = gate.BeginBoundary();

    complete_first();

    EXPECT_FALSE(gate.IsBoundaryOpen());
    complete_second();
    EXPECT_TRUE(gate.IsBoundaryOpen());
}

TEST(GnmSubmissionGate, HostBackpressureDoesNotTellTheGuestToDropAFrame) {
    SubmissionGate gate;
    auto complete_boundary = gate.BeginBoundary();

    EXPECT_FALSE(gate.IsBoundaryOpen());
    EXPECT_TRUE(gate.AreGuestSubmitsAllowed());

    complete_boundary();
}

TEST(GnmSubmissionTransaction, FlipRegistrationAndCommandEnqueueStayAtomic) {
    SubmissionGate gate;
    std::mutex trace_mutex;
    std::vector<std::string> trace;
    std::promise<void> first_registered;
    std::promise<void> release_first;
    auto first_registered_future = first_registered.get_future();
    auto release_first_future = release_first.get_future().share();

    const auto append = [&](std::string entry) {
        std::scoped_lock lock{trace_mutex};
        trace.emplace_back(std::move(entry));
    };

    auto first = std::async(std::launch::async, [&] {
        return RunSubmissionTransaction(
            gate, [] { return 0; },
            [&] {
                append("register A");
                first_registered.set_value();
                release_first_future.wait();
                return 0;
            },
            [&] {
                append("enqueue A");
                return 0;
            });
    });
    ASSERT_EQ(first_registered_future.wait_for(1s), std::future_status::ready);

    std::atomic<bool> second_validated{};
    auto second = std::async(std::launch::async, [&] {
        return RunSubmissionTransaction(
            gate,
            [&] {
                second_validated = true;
                return 0;
            },
            [&] {
                append("register B");
                return 0;
            },
            [&] {
                append("enqueue B");
                return 0;
            });
    });
    while (!second_validated.load()) {
        std::this_thread::yield();
    }

    EXPECT_EQ(second.wait_for(20ms), std::future_status::timeout);
    release_first.set_value();

    EXPECT_EQ(first.get(), 0);
    EXPECT_EQ(second.get(), 0);
    EXPECT_EQ(trace, (std::vector<std::string>{"register A", "enqueue A", "register B",
                                               "enqueue B"}));
}

TEST(GnmSubmissionTransaction, RejectedValidationLeavesNoFlipRegistration) {
    SubmissionGate gate;
    bool registered = false;
    bool enqueued = false;

    const auto result = RunSubmissionTransaction(
        gate, [] { return -1; },
        [&] {
            registered = true;
            return 0;
        },
        [&] {
            enqueued = true;
            return 0;
        });

    EXPECT_EQ(result, -1);
    EXPECT_FALSE(registered);
    EXPECT_FALSE(enqueued);
}

TEST(GnmSubmissionTransaction, RejectedFlipRegistrationLeavesNoCommandEnqueue) {
    SubmissionGate gate;
    bool enqueued = false;

    const auto result = RunSubmissionTransaction(
        gate, [] { return 0; }, [] { return -2; },
        [&] {
            enqueued = true;
            return 0;
        });

    EXPECT_EQ(result, -2);
    EXPECT_FALSE(enqueued);
}

} // namespace
} // namespace Libraries::GnmDriver
