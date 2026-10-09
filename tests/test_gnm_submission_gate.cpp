// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <ranges>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "core/libraries/gnmdriver/gfx_boundary_wait_diagnostic.h"
#include "core/libraries/gnmdriver/submission_gate.h"
#include "video_core/amdgpu/owned_command_buffers.h"
#include "video_core/amdgpu/submission_completion_queue.h"

using namespace std::chrono_literals;

namespace Libraries::GnmDriver {
namespace {

TEST(GfxBoundaryWaitDiagnostic, OnlyExactOneEnablesTheExperiment) {
    for (const char* value :
         std::array<const char*, 8>{nullptr, "", "0", "true", "01", "1 ", " 1", "-1"}) {
        EXPECT_FALSE(IsGfxBoundaryWaitDiagnosticEnabled(value));
    }
    EXPECT_TRUE(IsGfxBoundaryWaitDiagnosticEnabled("1"));
}

TEST(GfxBoundaryWaitDiagnostic, SelectsExactlyOneAdmissionPathWithoutTakingAnOuterGuard) {
    struct GateSpy {
        int entered{};
        int waited{};
        int Enter() {
            ++entered;
            return 11;
        }
        int WaitForBoundaryAndEnter() {
            ++waited;
            return 22;
        }
    } gate;
    EXPECT_EQ(EnterGraphicsSubmissionDiagnostic(gate, false), 11);
    EXPECT_EQ(gate.entered, 1);
    EXPECT_EQ(gate.waited, 0);
    EXPECT_EQ(EnterGraphicsSubmissionDiagnostic(gate, true), 22);
    EXPECT_EQ(gate.entered, 1);
    EXPECT_EQ(gate.waited, 1);
}

TEST(GfxBoundaryWaitDiagnostic, DisabledDoesNotWaitForAPendingParserBoundary) {
    SubmissionGate gate;
    auto complete = gate.BeginBoundary();
    auto graphics = std::async(std::launch::async, [&] {
        auto guard = EnterGraphicsSubmissionDiagnostic(gate, false);
        return guard.owns_lock();
    });
    const auto status = graphics.wait_for(1s);
    complete(); // Always rescue an accidentally waiting implementation before assertions/join.
    EXPECT_EQ(status, std::future_status::ready);
    EXPECT_TRUE(graphics.get());
}

TEST(GfxBoundaryWaitDiagnostic, OpenBoundaryEntersWithoutSelfDeadlock) {
    SubmissionGate gate;
    auto guard = EnterGraphicsSubmissionDiagnostic(gate, true);
    EXPECT_TRUE(guard.owns_lock());
    guard.unlock();
    EXPECT_TRUE(gate.IsBoundaryOpen());
}

TEST(GfxBoundaryWaitDiagnostic, PriorAscBoundaryCanBlockGfxWhileNewAscAdmissionRemainsOpen) {
    SubmissionGate gate;
    AmdGpu::SubmissionCompletionQueue queue;
    const auto prior_asc = queue.IssueSubmission();
    auto boundary = gate.BeginGuardedBoundary();
    queue.EnqueueBoundary(std::move(boundary.complete));
    boundary.issuance.unlock();
    std::promise<void> graphics_started;
    const bool gpu_fence_completed = false;
    auto graphics = std::async(std::launch::async, [&] {
        graphics_started.set_value();
        auto guard = EnterGraphicsSubmissionDiagnostic(gate, true);
        return std::pair{guard.owns_lock(), gpu_fence_completed};
    });
    graphics_started.get_future().wait();
    auto asc = std::async(std::launch::async, [&] {
        auto guard = gate.Enter(); // Actual unchanged DingDong admission path.
        return guard.owns_lock();
    });
    const auto asc_status = asc.wait_for(1s);
    const auto graphics_status = graphics.wait_for(20ms);

    // This is also an explicit limitation: if earlier ASC needs this next GFX, ON can
    // deadlock. Rescue the synthetic dependency before any assertion can leave a waiter.
    // Parser completion releases the gate even though no physical GPU fence completed.
    queue.CompleteSubmission(prior_asc);
    const auto graphics_result = graphics.get();
    EXPECT_EQ(asc_status, std::future_status::ready);
    EXPECT_TRUE(asc.get());
    EXPECT_EQ(graphics_status, std::future_status::timeout);
    EXPECT_TRUE(graphics_result.first);
    EXPECT_FALSE(graphics_result.second);
    EXPECT_TRUE(gate.AreGuestSubmitsAllowed());
    EXPECT_TRUE(gate.IsBoundaryOpen());
}

TEST(GfxBoundaryWaitDiagnostic, SourceWiringIsGfxOnlyAndProcessLocal) {
    const auto source_root =
        std::filesystem::path{LIVERPOOL_SOURCE_PATH}.parent_path().parent_path().parent_path();
    std::ifstream input{source_root / "core/libraries/gnmdriver/gnmdriver.cpp"};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto section = [&](std::string_view begin, std::string_view end) {
        const auto start = source.find(begin);
        const auto finish = source.find(end, start);
        return start == std::string::npos || finish == std::string::npos
                   ? std::string{}
                   : source.substr(start, finish - start);
    };
    const auto policy = section("static bool GfxBoundaryWaitDiagnosticEnabledForProcess()",
                                "static u64 frames_submitted");
    ASSERT_FALSE(policy.empty());
    EXPECT_NE(policy.find("static const bool enabled"), std::string::npos);
    EXPECT_NE(policy.find("IsGfxBoundaryWaitDiagnosticEnabled(std::getenv("), std::string::npos);
    EXPECT_NE(policy.find("SHADPS4_GFX_BOUNDARY_WAIT_DIAGNOSTIC"), std::string::npos);
    EXPECT_NE(policy.find("if (enabled)"), std::string::npos);
    const auto warning = policy.find("LOG_WARNING(");
    ASSERT_NE(warning, std::string::npos);
    EXPECT_EQ(policy.find("LOG_WARNING(", warning + 1), std::string::npos);

    const auto graphics = section("int PS4_SYSV_ABI sceGnmSubmitCommandBuffersForWorkload(",
                                  "s32 PS4_SYSV_ABI sceGnmSubmitCommandBuffers(");
    ASSERT_FALSE(graphics.empty());
    EXPECT_NE(graphics.find("EnterGraphicsSubmissionDiagnostic("), std::string::npos);
    EXPECT_NE(graphics.find("*submission_gate, GfxBoundaryWaitDiagnosticEnabledForProcess()"),
              std::string::npos);
    EXPECT_EQ(graphics.find("submission_gate->Enter()"), std::string::npos);
    const auto asc = section("void PS4_SYSV_ABI sceGnmDingDong(",
                             "void PS4_SYSV_ABI sceGnmDingDongForWorkload(");
    ASSERT_FALSE(asc.empty());
    EXPECT_NE(asc.find("submission_gate->Enter()"), std::string::npos);
    EXPECT_EQ(asc.find("EnterGraphicsSubmissionDiagnostic"), std::string::npos);
    const auto submit_done =
        section("int PS4_SYSV_ABI sceGnmSubmitDone()", "int PS4_SYSV_ABI sceGnmUnmapComputeQueue(");
    EXPECT_NE(submit_done.find("submission_gate->BeginGuardedBoundary()"), std::string::npos);
    EXPECT_EQ(submit_done.find("WaitForBoundaryAndEnter"), std::string::npos);
    EXPECT_EQ(submit_done.find("EnterGraphicsSubmissionDiagnostic"), std::string::npos);
    const auto eligibility = section("int PS4_SYSV_ABI sceGnmAreSubmitsAllowed()",
                                     "int PS4_SYSV_ABI sceGnmBeginWorkload(");
    EXPECT_NE(eligibility.find("submission_gate->AreGuestSubmitsAllowed()"), std::string::npos);
}

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

TEST(GnmSubmissionGate, GpuVisibleAddressChangeExcludesLaterSubmissions) {
    SubmissionGate gate;
    auto admission = gate.WaitForBoundaryAndEnter();
    std::promise<void> entrant_started;
    auto started = entrant_started.get_future();

    auto entrant = std::async(std::launch::async, [&] {
        entrant_started.set_value();
        auto submission = gate.Enter();
        return true;
    });
    ASSERT_EQ(started.wait_for(1s), std::future_status::ready);
    EXPECT_EQ(entrant.wait_for(20ms), std::future_status::timeout);

    admission.unlock();

    EXPECT_EQ(entrant.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(entrant.get());
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

TEST(GnmSubmissionGate, SubmitDoneAdmissionIsNotAnIssuedSubmissionCompletionFence) {
    SubmissionGate gate;
    AmdGpu::SubmissionCompletionQueue queue;
    const auto submitted = queue.IssueSubmission();
    bool issued_work_finished{};

    // No SubmitDone has been published. Its gate is open even though submitted work remains.
    // Holding this admission prevents later submissions; it does not finish the existing one.
    EXPECT_TRUE(gate.IsBoundaryOpen());
    auto admission = gate.WaitForBoundaryAndEnter();
    queue.EnqueueBoundary([&] { issued_work_finished = true; });
    queue.DrainReadyBoundaries();
    EXPECT_FALSE(issued_work_finished);

    queue.CompleteSubmission(submitted);
    EXPECT_TRUE(issued_work_finished);
}

TEST(SubmissionCompletionQueue, IssuedWorkFenceNeedsAllocatorProgressUntilPendingWorkCompletes) {
    AmdGpu::SubmissionCompletionQueue queue;
    const auto submitted = queue.IssueSubmission();
    std::promise<void> finished;
    auto finish = finished.get_future();
    queue.EnqueueBoundary([&] { finished.set_value(); });

    // Model an unrelated allocation needed to satisfy a pending guest WaitRegMem. A retirement
    // path must not hold the global allocation lock while waiting for all issued work to finish.
    std::mutex allocation_mutex;
    std::unique_lock retirement_allocation_lock{allocation_mutex};
    std::promise<void> allocation_started;
    auto started = allocation_started.get_future();
    auto allocation = std::async(std::launch::async, [&] {
        allocation_started.set_value();
        {
            std::scoped_lock lock{allocation_mutex};
            // Publishing fresh backing lets the guest satisfy its pending command dependency.
        }
        queue.CompleteSubmission(submitted);
    });

    EXPECT_EQ(started.wait_for(1s), std::future_status::ready);
    EXPECT_EQ(allocation.wait_for(20ms), std::future_status::timeout);
    EXPECT_EQ(finish.wait_for(0ms), std::future_status::timeout);

    // Always release before assertions that could end the test: the counterexample must never
    // leave an intentionally blocked worker behind.
    retirement_allocation_lock.unlock();
    EXPECT_EQ(allocation.wait_for(1s), std::future_status::ready);
    EXPECT_EQ(finish.wait_for(1s), std::future_status::ready);
    allocation.get();
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

std::array<u32, 4> MakeIndirectPacket(u32 opcode, std::span<const u32> commands) {
    const auto address = reinterpret_cast<uintptr_t>(commands.data());
    return {
        (3u << 30) | (2u << 16) | (opcode << 8),
        static_cast<u32>(address),
        static_cast<u32>(address >> 32) & 0xffffu,
        static_cast<u32>(commands.size()),
    };
}

TEST(OwnedCommandBuffers, SubmissionRecursivelyOwnsGraphicsAndConstantIndirectBuffers) {
    std::array<u32, 3> leaf{0x11111111, 0x22222222, 0x33333333};
    auto nested = MakeIndirectPacket(0x3f, leaf);
    auto dcb = MakeIndirectPacket(0x3f, nested);
    auto ccb = MakeIndirectPacket(0x33, leaf);
    const auto expected_nested = nested;

    const auto submission = AmdGpu::PrepareGraphicsSubmission(dcb, ccb);
    ASSERT_TRUE(submission.owner);
    const auto nested_address = reinterpret_cast<const u32*>(
        static_cast<uintptr_t>(dcb[1]) | (static_cast<uintptr_t>(dcb[2] & 0xffffu) << 32));
    const auto leaf_address = reinterpret_cast<const u32*>(
        static_cast<uintptr_t>(nested[1]) | (static_cast<uintptr_t>(nested[2] & 0xffffu) << 32));

    nested.fill(0);
    leaf.fill(0);

    EXPECT_TRUE(
        std::ranges::equal(submission.owner->ResolveIndirect(nested_address, 4), expected_nested));
    EXPECT_TRUE(std::ranges::equal(submission.owner->ResolveIndirect(leaf_address, 3),
                                   std::array<u32, 3>{0x11111111, 0x22222222, 0x33333333}));
}

TEST(OwnedCommandBuffers, LiverpoolResolvesBothIndirectPacketKindsThroughSubmissionOwner) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};

    const auto first_resolve = source.find("ResolveIndirect");
    ASSERT_NE(first_resolve, std::string::npos);
    EXPECT_NE(source.find("ResolveIndirect", first_resolve + 1), std::string::npos);
    EXPECT_NE(source.find("ProcessGraphics(indirect_commands"), std::string::npos);
    EXPECT_NE(source.find("ProcessCeUpdate(indirect_commands"), std::string::npos);
}

} // namespace
} // namespace Libraries::GnmDriver
