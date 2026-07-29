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

TEST(GnmSubmissionGate, SubmitDoneClosesIdleGateUntilGpuAcknowledgesIt) {
    SubmissionGate gate;

    auto complete_boundary = gate.BeginBoundary();

    EXPECT_FALSE(gate.IsOpen());

    auto entrant = std::async(std::launch::async, [&] {
        auto guard = gate.Enter();
        return true;
    });
    EXPECT_EQ(entrant.wait_for(20ms), std::future_status::timeout);

    complete_boundary();

    EXPECT_EQ(entrant.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(entrant.get());
}

TEST(GnmSubmissionGate, SubmitDoneCannotSplitAnInProgressSubmission) {
    SubmissionGate gate;
    auto submission = gate.Enter();

    std::atomic<bool> close_started{};
    auto closer = std::async(std::launch::async, [&] {
        close_started = true;
        auto complete_boundary = gate.BeginBoundary();
        return complete_boundary;
    });
    while (!close_started.load()) {
        std::this_thread::yield();
    }

    EXPECT_EQ(closer.wait_for(20ms), std::future_status::timeout);

    submission = {};

    EXPECT_EQ(closer.wait_for(1s), std::future_status::ready);
    auto complete_boundary = closer.get();
    EXPECT_FALSE(gate.IsOpen());
    complete_boundary();
}

TEST(GnmSubmissionGate, ConsecutiveSubmitDoneBoundariesDoNotCoalesce) {
    SubmissionGate gate;
    auto complete_first_boundary = gate.BeginBoundary();

    std::atomic<bool> second_close_started{};
    auto second_closer = std::async(std::launch::async, [&] {
        second_close_started = true;
        return gate.BeginBoundary();
    });
    while (!second_close_started.load()) {
        std::this_thread::yield();
    }

    EXPECT_EQ(second_closer.wait_for(20ms), std::future_status::timeout);

    complete_first_boundary();

    EXPECT_EQ(second_closer.wait_for(1s), std::future_status::ready);
    auto complete_second_boundary = second_closer.get();
    EXPECT_FALSE(gate.IsOpen());
    complete_second_boundary();
}

TEST(GnmSubmissionGate, StaleCompletionCannotAcknowledgeANewerBoundary) {
    SubmissionGate gate;

    auto complete_first_boundary = gate.BeginBoundary();
    complete_first_boundary();
    auto complete_second_boundary = gate.BeginBoundary();

    complete_first_boundary();

    EXPECT_FALSE(gate.IsOpen());
    complete_second_boundary();
    EXPECT_TRUE(gate.IsOpen());
}

} // namespace
} // namespace Libraries::GnmDriver
