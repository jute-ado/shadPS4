// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include <gtest/gtest.h>

#include "core/libraries/gnmdriver/submission_gate.h"

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

TEST(GnmSubmissionGate, SubmitDoneReturnWaitsForCommandProcessorBoundary) {
    SubmissionGate gate;
    std::promise<Common::UniqueFunction<void>> queued_completion;
    auto completion = queued_completion.get_future();

    auto submit_done = std::async(std::launch::async, [&] {
        gate.SubmitBoundaryAndWait([&](Common::UniqueFunction<void>&& complete_boundary) {
            queued_completion.set_value(std::move(complete_boundary));
        });
        return true;
    });

    auto complete_boundary = completion.get();
    EXPECT_EQ(submit_done.wait_for(20ms), std::future_status::timeout);

    complete_boundary();

    EXPECT_EQ(submit_done.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(submit_done.get());
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

} // namespace
} // namespace Libraries::GnmDriver
