// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <ranges>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "core/libraries/gnmdriver/submission_gate.h"
#include "video_core/amdgpu/owned_command_buffers.h"
#include "video_core/amdgpu/submission_completion_queue.h"

using namespace std::chrono_literals;

namespace Libraries::GnmDriver {
namespace {

TEST(GnmSubmissionGate, PendingBoundaryDoesNotBlockTheNextSubmission) {
    SubmissionGate gate;
    auto complete_boundary = gate.BeginBoundary();

    auto entrant = std::async(std::launch::async, [&] {
        auto guard = gate.Enter();
        return true;
    });
    const auto status = entrant.wait_for(20ms);
    complete_boundary();

    EXPECT_EQ(status, std::future_status::ready);
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

TEST(GnmSubmissionGate, PublishedBoundaryCannotBeOvertakenByTheNextSubmission) {
    SubmissionGate gate;
    auto boundary = gate.BeginGuardedBoundary();

    auto entrant = std::async(std::launch::async, [&] {
        auto guard = gate.Enter();
        return true;
    });
    EXPECT_EQ(entrant.wait_for(20ms), std::future_status::timeout);

    boundary.issuance.unlock();

    EXPECT_EQ(entrant.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(entrant.get());
    boundary.complete();
}

TEST(GnmSubmissionGate, ConsecutiveBoundariesDoNotBlockTheGuestAndWaitForAllCompletions) {
    SubmissionGate gate;
    auto complete_first = gate.BeginBoundary();

    auto second_boundary = std::async(std::launch::async, [&] { return gate.BeginBoundary(); });
    const auto second_status = second_boundary.wait_for(20ms);

    complete_first();
    auto complete_second = second_boundary.get();

    EXPECT_EQ(second_status, std::future_status::ready);
    EXPECT_FALSE(gate.IsBoundaryOpen());

    auto waiter = std::async(std::launch::async, [&] {
        gate.WaitForBoundary();
        return true;
    });
    EXPECT_EQ(waiter.wait_for(20ms), std::future_status::timeout);

    complete_second();
    EXPECT_EQ(waiter.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(waiter.get());
    EXPECT_TRUE(gate.IsBoundaryOpen());
}

TEST(SubmissionCompletionQueue, ReadyBoundaryIsNotStarvedByLaterSubmissions) {
    AmdGpu::SubmissionCompletionQueue queue;
    const auto before_boundary = queue.IssueSubmission();
    bool boundary_completed{};
    queue.EnqueueBoundary([&] { boundary_completed = true; });
    const auto after_boundary = queue.IssueSubmission();

    queue.CompleteSubmission(before_boundary);

    EXPECT_TRUE(boundary_completed);
    EXPECT_FALSE(queue.HasPendingBoundaries());

    queue.CompleteSubmission(after_boundary);
}

TEST(SubmissionCompletionQueue, OutOfOrderSubmissionCompletionPreservesBoundaryOrder) {
    AmdGpu::SubmissionCompletionQueue queue;
    const auto first = queue.IssueSubmission();
    const auto second = queue.IssueSubmission();
    std::vector<u32> completed_boundaries;
    queue.EnqueueBoundary([&] { completed_boundaries.push_back(1); });

    queue.CompleteSubmission(second);
    EXPECT_TRUE(completed_boundaries.empty());

    queue.CompleteSubmission(first);
    EXPECT_EQ(completed_boundaries, std::vector<u32>{1});
}

TEST(SubmissionCompletionQueue, MultipleBoundariesRetireIndependently) {
    AmdGpu::SubmissionCompletionQueue queue;
    const auto first = queue.IssueSubmission();
    std::vector<u32> completed_boundaries;
    queue.EnqueueBoundary([&] { completed_boundaries.push_back(1); });
    const auto second = queue.IssueSubmission();
    queue.EnqueueBoundary([&] { completed_boundaries.push_back(2); });

    queue.CompleteSubmission(first);
    EXPECT_EQ(completed_boundaries, std::vector<u32>{1});
    EXPECT_TRUE(queue.HasPendingBoundaries());

    queue.CompleteSubmission(second);
    EXPECT_EQ(completed_boundaries, (std::vector<u32>{1, 2}));
    EXPECT_FALSE(queue.HasPendingBoundaries());
}

TEST(SubmissionCompletionQueue, BoundaryWithoutSubmissionsRunsOnExplicitDrain) {
    AmdGpu::SubmissionCompletionQueue queue;
    bool boundary_completed{};
    queue.EnqueueBoundary([&] { boundary_completed = true; });

    EXPECT_FALSE(boundary_completed);
    queue.DrainReadyBoundaries();

    EXPECT_TRUE(boundary_completed);
    EXPECT_FALSE(queue.HasPendingBoundaries());
}

TEST(OwnedCommandBuffers, ConsecutiveCopiesRetainIndependentImmutableStorage) {
    std::array<u32, 3> first_dcb{1, 2, 3};
    std::array<u32, 2> first_ccb{4, 5};
    auto first = AmdGpu::OwnedCommandBuffers::Copy(first_dcb, first_ccb);

    std::array<u32, 2> second_dcb{6, 7};
    std::array<u32, 1> second_ccb{8};
    auto second = AmdGpu::OwnedCommandBuffers::Copy(second_dcb, second_ccb);

    first_dcb.fill(0);
    first_ccb.fill(0);
    second_dcb.fill(0);
    second_ccb.fill(0);

    EXPECT_TRUE(std::ranges::equal(first->Dcb(), std::array<u32, 3>{1, 2, 3}));
    EXPECT_TRUE(std::ranges::equal(first->Ccb(), std::array<u32, 2>{4, 5}));
    EXPECT_TRUE(std::ranges::equal(second->Dcb(), std::array<u32, 2>{6, 7}));
    EXPECT_TRUE(std::ranges::equal(second->Ccb(), std::array<u32, 1>{8}));
    EXPECT_NE(first->Dcb().data(), second->Dcb().data());
    EXPECT_NE(first->Ccb().data(), second->Ccb().data());
}

TEST(OwnedCommandBuffers, GraphicsSubmissionAlwaysTakesImmutableOwnership) {
    std::array<u32, 3> dcb{0x10, 0x20, 0x30};
    std::array<u32, 2> ccb{0x40, 0x50};

    const auto submission = AmdGpu::PrepareGraphicsSubmission(dcb, ccb);
    ASSERT_TRUE(submission.owner);
    EXPECT_NE(submission.dcb.data(), dcb.data());
    EXPECT_NE(submission.ccb.data(), ccb.data());

    dcb.fill(0);
    ccb.fill(0);

    EXPECT_TRUE(std::ranges::equal(submission.dcb, std::array<u32, 3>{0x10, 0x20, 0x30}));
    EXPECT_TRUE(std::ranges::equal(submission.ccb, std::array<u32, 2>{0x40, 0x50}));
}

} // namespace
} // namespace Libraries::GnmDriver
