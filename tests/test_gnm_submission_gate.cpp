// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
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
#include "core/libraries/gnmdriver/submission_transaction.h"
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

TEST(GnmSubmissionTransaction, FlipSourceValidatesAndPublishesUnderOneAdmission) {
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

    std::ifstream transaction_input{source_root /
                                    "core/libraries/gnmdriver/submission_transaction.h"};
    ASSERT_TRUE(transaction_input);
    const std::string transaction_source{std::istreambuf_iterator<char>{transaction_input}, {}};
    const auto transaction_header =
        std::string_view{"template <typename Validate, typename Register, typename Enqueue>"};
    const auto transaction_start = transaction_source.find(transaction_header);
    const auto transaction_end = transaction_source.find("} // namespace Libraries::GnmDriver",
                                                        transaction_start);
    const auto transaction = transaction_start == std::string::npos ||
                                     transaction_end == std::string::npos
                                 ? std::string{}
                                 : transaction_source.substr(transaction_start,
                                                             transaction_end - transaction_start);
    ASSERT_FALSE(transaction.empty());
    const auto validation = transaction.find("validation_result");
    const auto admission = transaction.find("auto submission = gate.Enter()");
    const auto registration = transaction.find("registration_result");
    ASSERT_NE(validation, std::string::npos);
    ASSERT_NE(admission, std::string::npos);
    ASSERT_NE(registration, std::string::npos);
    EXPECT_LT(validation, admission);
    EXPECT_LT(admission, registration);

    const auto flip = section("s32 PS4_SYSV_ABI sceGnmSubmitAndFlipCommandBuffersForWorkload(",
                              "static int SubmitCommandBuffersLocked(");
    ASSERT_FALSE(flip.empty());
    EXPECT_NE(flip.find("ValidateCommandBuffers("), std::string::npos);
    EXPECT_NE(flip.find("SubmitCommandBuffersLocked("), std::string::npos);
    EXPECT_NE(flip.find("PatchFlipRequest("), std::string::npos);
    EXPECT_EQ(flip.find("sceGnmSubmitCommandBuffers("), std::string::npos);

    const auto validator = section("static int ValidateCommandBuffers(",
                                   "static int SubmitCommandBuffersLocked(");
    ASSERT_FALSE(validator.empty());
    EXPECT_NE(validator.find("count == 0"), std::string::npos);
    EXPECT_NE(validator.find("dcb_gpu_addrs == nullptr"), std::string::npos);
    EXPECT_NE(validator.find("dcb_sizes_in_bytes == nullptr"), std::string::npos);
    EXPECT_NE(validator.find("dcb_gpu_addrs[i] == nullptr"), std::string::npos);
    EXPECT_NE(validator.find("IsReadableGuestArray"), std::string::npos);
    EXPECT_NE(validator.find("sizeof(u32)"), std::string::npos);
    EXPECT_NE(source.find("memory->CanReadBacking"), std::string::npos);
    EXPECT_NE(validator.find("dcb_sizes_in_bytes[count - 1] < MinimumFlipCommandBufferSizeInBytes"),
              std::string::npos);
    EXPECT_NE(validator.find("buf_idx >= VideoOut::MaxDisplayBuffers"), std::string::npos);
}

TEST(GnmSubmissionTransaction, EopFlipRejectsClosedInvalidAndUnregisteredBuffers) {
    const auto source_root =
        std::filesystem::path{LIVERPOOL_SOURCE_PATH}.parent_path().parent_path().parent_path();
    std::ifstream input{source_root / "core/libraries/videoout/video_out.cpp"};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto begin = source.find("s32 sceVideoOutSubmitEopFlip(");
    const auto end = source.find("\ns32 PS4_SYSV_ABI sceVideoOutGetDeviceCapabilityInfo(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    EXPECT_NE(body.find("!port"), std::string::npos);
    EXPECT_NE(body.find("port->lifecycle.IsOpen()"), std::string::npos);
    EXPECT_NE(body.find("buf_id >= MaxDisplayBuffers"), std::string::npos);
    EXPECT_NE(body.find("port->buffer_slots[buf_id].group_index < 0"), std::string::npos);
    EXPECT_NE(body.find("RegisterOnce"), std::string::npos);
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

TEST(GnmSubmissionGate, CompletionCanAcknowledgeWhileAdmissionIsRetained) {
    SubmissionGate gate;
    auto boundary = gate.BeginGuardedBoundary();
    boundary.issuance.unlock();

    // Model a GPU-visible unmap holding submission admission while a synchronous owner-side
    // readback waits for the worker to run the already-published SubmitDone callback.
    auto admission = gate.Enter();
    auto completion = std::async(std::launch::async, [&] {
        boundary.complete();
        return gate.IsBoundaryOpen();
    });

    const auto status = completion.wait_for(1s);
    // Always release the modeled unmap admission before returning, even when the regression is
    // present, so a failing implementation cannot strand the async worker in future destruction.
    if (status != std::future_status::ready) {
        admission.unlock();
    }
    ASSERT_EQ(status, std::future_status::ready);
    EXPECT_TRUE(completion.get());
    if (admission.owns_lock()) {
        admission.unlock();
    }
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

TEST(OwnedCommandBuffers, ComputeSubmissionAlwaysTakesImmutableOwnership) {
    std::array<u32, 4> acb{0x10, 0x20, 0x30, 0x40};

    const auto submission = AmdGpu::PrepareComputeSubmission(acb);
    ASSERT_TRUE(submission.owner);
    EXPECT_NE(submission.acb.data(), acb.data());

    acb.fill(0);

    EXPECT_TRUE(std::ranges::equal(submission.acb, std::array<u32, 4>{0x10, 0x20, 0x30, 0x40}));
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

TEST(OwnedCommandBuffers, ComputeSubmissionRecursivelyOwnsIndirectBuffers) {
    std::array<u32, 3> leaf{0x11111111, 0x22222222, 0x33333333};
    auto root = MakeIndirectPacket(0x3f, leaf);
    const auto submission = AmdGpu::PrepareComputeSubmission(root);
    ASSERT_TRUE(submission.owner);

    const auto leaf_address = reinterpret_cast<const u32*>(
        static_cast<uintptr_t>(root[1]) | (static_cast<uintptr_t>(root[2] & 0xffffu) << 32));
    leaf.fill(0);

    EXPECT_TRUE(std::ranges::equal(submission.owner->ResolveIndirect(leaf_address, 3),
                                   std::array<u32, 3>{0x11111111, 0x22222222, 0x33333333}));
}

TEST(OwnedCommandBuffers, SplitRingIndirectPacketIsCapturedAfterReassembly) {
    std::array<u32, 3> leaf{0x11111111, 0x22222222, 0x33333333};
    const auto packet = MakeIndirectPacket(0x3f, leaf);

    // The ring producer submitted the packet header/address-low in the tail and the remaining
    // address/size words in the wrapped head. Capturing either span alone cannot discover the IB.
    const std::array<u32, 2> tail{packet[0], packet[1]};
    const std::array<u32, 2> head{packet[2], packet[3]};
    const std::array<u32, 4> reassembled{tail[0], tail[1], head[0], head[1]};
    const auto submission = AmdGpu::OwnedCommandBuffers::CopyWithIndirectPacket(
        head, {}, reassembled);
    ASSERT_TRUE(submission);

    const auto leaf_address = reinterpret_cast<const u32*>(
        static_cast<uintptr_t>(packet[1]) |
        (static_cast<uintptr_t>(packet[2] & 0xffffu) << 32));
    leaf.fill(0);

    EXPECT_TRUE(std::ranges::equal(submission->ResolveIndirect(leaf_address, 3),
                                   std::array<u32, 3>{0x11111111, 0x22222222, 0x33333333}));
}

TEST(OwnedCommandBuffers, UnownedIndirectBufferFailsClosedInsteadOfBorrowingGuestPointer) {
    const auto submission = AmdGpu::OwnedCommandBuffers::Copy({}, {});
    ASSERT_TRUE(submission);

    // A deferred parser must never borrow an address that was not captured with the submission.
    // Use a deliberately invalid sentinel: the regression test must not dereference it merely
    // to decide whether the owner can resolve the indirect buffer.
    const auto* unowned_address = reinterpret_cast<const u32*>(uintptr_t{0x1});
    EXPECT_TRUE(submission->ResolveIndirect(unowned_address, 4).empty());
}

TEST(OwnedCommandBuffers, GuestCopyRejectsUnreadableRootWithoutDereferencingIt) {
    const auto* unreadable_root = reinterpret_cast<const u32*>(uintptr_t{0x1});
    u32 reader_calls{};
    const AmdGpu::GuestCommandBufferReader reader = [&](VAddr, std::span<u32>) {
        ++reader_calls;
        return false;
    };

    const auto owner = AmdGpu::OwnedCommandBuffers::CopyFromGuest(
        std::span<const u32>{unreadable_root, 4}, {}, reader);
    EXPECT_FALSE(owner);
    EXPECT_EQ(reader_calls, 1U);
}

TEST(OwnedCommandBuffers, GuestCopyReadsIndirectBuffersThroughTheVmmReader) {
    std::array<u32, 3> leaf{0x11111111, 0x22222222, 0x33333333};
    auto root = MakeIndirectPacket(0x3f, leaf);
    const auto leaf_address = reinterpret_cast<const u32*>(
        static_cast<uintptr_t>(root[1]) | (static_cast<uintptr_t>(root[2] & 0xffffu) << 32));
    const auto root_address = reinterpret_cast<VAddr>(root.data());
    const auto reader = [&](VAddr address, std::span<u32> destination) {
        if (address == root_address) {
            std::memcpy(destination.data(), root.data(), destination.size_bytes());
            return true;
        }
        if (address == reinterpret_cast<VAddr>(leaf.data())) {
            std::memcpy(destination.data(), leaf.data(), destination.size_bytes());
            return true;
        }
        return false;
    };

    const auto owner = AmdGpu::OwnedCommandBuffers::CopyFromGuest(root, {}, reader);
    ASSERT_TRUE(owner);
    root.fill(0);
    leaf.fill(0);

    EXPECT_TRUE(std::ranges::equal(owner->ResolveIndirect(leaf_address, 3),
                                   std::array<u32, 3>{0x11111111, 0x22222222, 0x33333333}));
}

TEST(OwnedCommandBuffers, GuestCopyDropsUnreadableIndirectBufferWithoutRawDereference) {
    std::array<u32, 4> root{
        (3u << 30) | (2u << 16) | (0x3fu << 8),
        0x1u,
        0u,
        4u,
    };
    const auto root_address = reinterpret_cast<VAddr>(root.data());
    u32 reader_calls{};
    const auto reader = [&](VAddr address, std::span<u32> destination) {
        ++reader_calls;
        if (address != root_address) {
            return false;
        }
        std::memcpy(destination.data(), root.data(), destination.size_bytes());
        return true;
    };

    const auto owner = AmdGpu::OwnedCommandBuffers::CopyFromGuest(root, {}, reader);
    ASSERT_TRUE(owner);
    EXPECT_EQ(reader_calls, 2U);
    EXPECT_TRUE(owner->ResolveIndirect(reinterpret_cast<const u32*>(uintptr_t{0x1}), 4).empty());
}

TEST(AscSubmissionLifetime, DeferredComputeCarriesOwnedCommandAndQueueState) {
    std::ifstream liverpool_input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(liverpool_input);
    const std::string liverpool{std::istreambuf_iterator<char>{liverpool_input}, {}};
    std::ifstream gnm_input{SHADPS4_GNM_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(gnm_input);
    const std::string gnm{std::istreambuf_iterator<char>{gnm_input}, {}};

    const auto process_begin = liverpool.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto process_end = liverpool.find("void Liverpool::SubmitGfx(", process_begin);
    ASSERT_NE(process_begin, std::string::npos);
    ASSERT_NE(process_end, std::string::npos);
    const auto process = liverpool.substr(process_begin, process_end - process_begin);

    const auto submit_begin = liverpool.find("void Liverpool::SubmitAsc(");
    const auto submit_end = liverpool.find("void Liverpool::SubmitDone(", submit_begin);
    ASSERT_NE(submit_begin, std::string::npos);
    ASSERT_NE(submit_end, std::string::npos);
    const auto submit = liverpool.substr(submit_begin, submit_end - submit_begin);

    const auto ding_begin = gnm.find("void PS4_SYSV_ABI sceGnmDingDong(");
    const auto ding_end = gnm.find("void PS4_SYSV_ABI sceGnmDingDongForWorkload(", ding_begin);
    ASSERT_NE(ding_begin, std::string::npos);
    ASSERT_NE(ding_end, std::string::npos);
    const auto ding = gnm.substr(ding_begin, ding_end - ding_begin);

    EXPECT_NE(submit.find("PrepareComputeSubmission"), std::string::npos);
    EXPECT_NE(submit.find("ProcessCompute(owned_submission.acb"), std::string::npos);
    EXPECT_NE(process.find("std::shared_ptr<AscQueueInfo>"), std::string::npos);
    EXPECT_NE(process.find("owner->ResolveIndirect"), std::string::npos);
    EXPECT_EQ(process.find("asc_queues[{vqid}]"), std::string::npos);
    EXPECT_NE(ding.find("PrepareComputeSubmission"), std::string::npos);
}

TEST(AscSubmissionLifetime, UnmapRetiresQueueGenerationBeforeSlotErase) {
    std::ifstream input{SHADPS4_GNM_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto begin = source.find("int PS4_SYSV_ABI sceGnmUnmapComputeQueue(");
    const auto end = source.find("int PS4_SYSV_ABI sceGnmUnregisterAllResourcesForOwner(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    const auto admission = body.find("submission_gate->Enter()");
    const auto lock = body.find("asc_queue_mutex");
    const auto invalidate = body.find("retired.store(true");
    const auto erase = body.find("asc_queues.erase(");
    ASSERT_NE(admission, std::string::npos);
    ASSERT_NE(lock, std::string::npos);
    ASSERT_NE(invalidate, std::string::npos);
    ASSERT_NE(erase, std::string::npos);
    EXPECT_LT(admission, lock);
    EXPECT_LT(lock, invalidate);
    EXPECT_LT(invalidate, erase);
}

TEST(AscSubmissionLifetime, RetiringQueueDoesNotCancelAlreadyAdmittedPackets) {
    struct QueueState {
        std::atomic_bool retired{false};
        std::atomic_bool active{true};
    } queue;
    std::array<u32, 3> admitted{1, 2, 3};
    std::vector<u32> executed;

    // Unmap retires admission, but a command buffer captured before that point remains eligible
    // for execution. This model prevents a lifecycle bit from silently dropping startup work.
    queue.retired.store(true, std::memory_order_release);
    if (queue.active.load(std::memory_order_acquire)) {
        executed.assign(admitted.begin(), admitted.end());
    }
    EXPECT_TRUE(queue.retired.load(std::memory_order_acquire));
    EXPECT_EQ(executed, (std::vector<u32>{1, 2, 3}));

    const auto source_root =
        std::filesystem::path{LIVERPOOL_SOURCE_PATH}.parent_path().parent_path().parent_path();
    std::ifstream liverpool_input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(liverpool_input);
    const std::string liverpool{std::istreambuf_iterator<char>{liverpool_input}, {}};
    const auto process_begin = liverpool.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto process_end = liverpool.find("void Liverpool::SubmitGfx(", process_begin);
    ASSERT_NE(process_begin, std::string::npos);
    ASSERT_NE(process_end, std::string::npos);
    const auto process = liverpool.substr(process_begin, process_end - process_begin);
    EXPECT_EQ(process.find("while (!acb.empty() && queue_state->active.load(std::memory_order_acquire) &&\n                           !queue_state->retired.load"),
              std::string::npos);
    EXPECT_NE(process.find("!queue_state->retired.load(std::memory_order_acquire)"),
              std::string::npos);

    std::ifstream gnm_input{source_root / "core/libraries/gnmdriver/gnmdriver.cpp",
                            std::ios::binary};
    ASSERT_TRUE(gnm_input);
    const std::string gnm{std::istreambuf_iterator<char>{gnm_input}, {}};
    const auto ding_begin = gnm.find("void PS4_SYSV_ABI sceGnmDingDong(");
    const auto ding_end = gnm.find("void PS4_SYSV_ABI sceGnmDingDongForWorkload(", ding_begin);
    const auto unmap_begin = gnm.find("int PS4_SYSV_ABI sceGnmUnmapComputeQueue(");
    const auto unmap_end =
        gnm.find("int PS4_SYSV_ABI sceGnmUnregisterAllResourcesForOwner(", unmap_begin);
    ASSERT_NE(ding_begin, std::string::npos);
    ASSERT_NE(ding_end, std::string::npos);
    ASSERT_NE(unmap_begin, std::string::npos);
    ASSERT_NE(unmap_end, std::string::npos);
    EXPECT_NE(gnm.substr(ding_begin, ding_end - ding_begin).find("retired.load"),
              std::string::npos);
    EXPECT_NE(gnm.substr(unmap_begin, unmap_end - unmap_begin).find("retired.store(true"),
              std::string::npos);
}

TEST(AscSubmissionLifetime, ReadPointerPublicationUsesCheckedGuestMemory) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto begin = source.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto end = source.find("void Liverpool::SubmitGfx(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_NE(body.find("TryWriteGuest"), std::string::npos);
    EXPECT_EQ(body.find("*queue.read_addr"), std::string::npos);
}

TEST(AscSubmissionLifetime, DeferredGuestMemoryOpsUseCheckedMemoryAccess) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto begin = source.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto end = source.find("void Liverpool::SubmitGfx(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    // ASC packets execute after their producer returns. Guest addresses must therefore go
    // through the VMA-checked API instead of being reinterpreted as process pointers.
    EXPECT_NE(body.find("TryReadBacking"), std::string::npos);
    EXPECT_NE(body.find("TryWriteGuest"), std::string::npos);
    EXPECT_EQ(body.find("write_data->Address<void*>()"), std::string::npos);
    EXPECT_EQ(body.find("mem_semaphore->Signal()"), std::string::npos);
    EXPECT_EQ(body.find("mem_semaphore->Signaled()"), std::string::npos);
    EXPECT_EQ(body.find("mem_semaphore->Decrement()"), std::string::npos);
    EXPECT_EQ(body.find("wait_reg_mem->Test(regs.reg_array)"), std::string::npos);
    EXPECT_EQ(body.find("release_mem->SignalFence("), std::string::npos);
}

TEST(GraphicsSubmissionLifetime, DeferredGuestMemoryOpsUseCheckedMemoryAccess) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto graphics_begin = source.find("Liverpool::Task Liverpool::ProcessGraphics(");
    const auto graphics_end = source.find("Liverpool::Task Liverpool::ProcessCompute(",
                                          graphics_begin);
    ASSERT_NE(graphics_begin, std::string::npos);
    ASSERT_NE(graphics_end, std::string::npos);
    const auto graphics = source.substr(graphics_begin, graphics_end - graphics_begin);

    // Graphics packets are also deferred past guest submission. Any guest VAddr touched by the
    // parser must use the VMA-checked API rather than becoming a host pointer and being
    // dereferenced on the GPU command thread.
    EXPECT_NE(graphics.find("TryReadBacking"), std::string::npos);
    EXPECT_NE(graphics.find("TryWriteGuest"), std::string::npos);
    EXPECT_NE(graphics.find("packet_words > dcb.size()"), std::string::npos);

    const auto section = [&](std::string_view begin, std::string_view end) {
        const auto first = graphics.find(begin);
        const auto last = graphics.find(end, first == std::string::npos ? 0 : first + begin.size());
        return first == std::string::npos || last == std::string::npos
                   ? std::string_view{}
                   : std::string_view{graphics}.substr(first, last - first);
    };

    const auto event_write = section("case PM4ItOpcode::EventWrite:",
                                     "case PM4ItOpcode::EventWriteEos:");
    ASSERT_FALSE(event_write.empty());
    EXPECT_NE(event_write.find("try_write_guest"), std::string::npos);
    EXPECT_EQ(event_write.find("event->Address<u64*>()"), std::string::npos);
    EXPECT_EQ(event_write.find("*results"), std::string::npos);

    const auto write_data = section("case PM4ItOpcode::WriteData:",
                                    "case PM4ItOpcode::CopyData:");
    ASSERT_FALSE(write_data.empty());
    EXPECT_NE(write_data.find("try_write_guest"), std::string::npos);
    EXPECT_EQ(write_data.find("write_data->Address<u64*>()"), std::string::npos);
    EXPECT_EQ(write_data.find("memcpy(address"), std::string::npos);

    const auto mem_semaphore = section("case PM4ItOpcode::MemSemaphore:",
                                        "case PM4ItOpcode::AcquireMem:");
    ASSERT_FALSE(mem_semaphore.empty());
    EXPECT_NE(mem_semaphore.find("try_read_guest"), std::string::npos);
    EXPECT_NE(mem_semaphore.find("try_write_guest"), std::string::npos);
    EXPECT_EQ(mem_semaphore.find("mem_semaphore->Signal()"), std::string::npos);
    EXPECT_EQ(mem_semaphore.find("mem_semaphore->Signaled()"), std::string::npos);
    EXPECT_EQ(mem_semaphore.find("mem_semaphore->Decrement()"), std::string::npos);

    const auto wait_reg_mem = section("case PM4ItOpcode::WaitRegMem:",
                                      "case PM4ItOpcode::IndirectBuffer:");
    ASSERT_FALSE(wait_reg_mem.empty());
    EXPECT_NE(wait_reg_mem.find("test_wait_reg_mem"), std::string::npos);
    // The internal video-output label path must also poll cooperatively so the graphics owner can
    // service the EOP flip that releases the label; ordinary waits use test_wait_reg_mem too.
    EXPECT_EQ(wait_reg_mem.find("vo_port->WaitVoLabel"), std::string::npos);
    EXPECT_NE(wait_reg_mem.find("while (vo_port->lifecycle.IsCurrent(wait_lifecycle_epoch)"),
              std::string::npos);
    EXPECT_NE(wait_reg_mem.find("YIELD_GFX();"), std::string::npos);

    const auto cond_exec = section("case PM4ItOpcode::CondExec:", "default:");
    ASSERT_FALSE(cond_exec.empty());
    EXPECT_NE(cond_exec.find("try_read_guest"), std::string::npos);
    EXPECT_EQ(cond_exec.find("*cond_exec->Address()"), std::string::npos);
}

TEST(GraphicsSubmissionLifetime, WriteDataUsesBodyLengthAndZeroSizeGuard) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};

    const auto first = source.find("case PM4ItOpcode::WriteData:");
    ASSERT_NE(first, std::string::npos);
    const auto second = source.find("case PM4ItOpcode::WriteData:", first + 1);
    ASSERT_NE(second, std::string::npos);

    const auto check_section = [&](size_t begin, std::string_view end_marker) {
        const auto end = source.find(end_marker, begin);
        EXPECT_NE(end, std::string::npos);
        if (end == std::string::npos) {
            return std::string_view{};
        }
        return std::string_view{source}.substr(begin, end - begin);
    };

    const auto graphics = check_section(first, "case PM4ItOpcode::CopyData:");
    const auto asc = check_section(second, "case PM4ItOpcode::MemSemaphore:");
    ASSERT_FALSE(graphics.empty());
    ASSERT_FALSE(asc.empty());

    // NumWords() is the body length, while COUNT is N - 1. WRITE_DATA reserves three body
    // dwords (control plus the 64-bit destination); computing from COUNT after switching to
    // NumWords() would read one dword past a complete packet. A zero-payload packet must also
    // remain a no-op instead of entering the host-owned VO-label path with a null owner.
    for (const auto section : {graphics, asc}) {
        EXPECT_NE(
            section.find("const auto body_words = static_cast<size_t>(header->type3.NumWords());"),
            std::string_view::npos);
        EXPECT_NE(section.find("body_words < 3"), std::string_view::npos);
        EXPECT_NE(section.find("(body_words - 3) * sizeof(u32)"), std::string_view::npos);
    }

    const auto graphics_begin = source.find("Liverpool::Task Liverpool::ProcessGraphics(");
    const auto compute_begin = source.find("Liverpool::Task Liverpool::ProcessCompute(",
                                           graphics_begin);
    ASSERT_NE(graphics_begin, std::string::npos);
    ASSERT_NE(compute_begin, std::string::npos);
    const auto graphics_body =
        std::string_view{source}.substr(graphics_begin, compute_begin - graphics_begin);
    const auto zero_guard = graphics_body.find("if (size == 0) {");
    ASSERT_NE(zero_guard, std::string_view::npos);
    EXPECT_LT(zero_guard, graphics_body.find("vo_port->vo_mutex"));
}

TEST(GraphicsSubmissionLifetime, PredicationBoundsPacketBeforeSyntaxDecode) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};

    const auto begin = source.find("case PM4ItOpcode::SetPredication:");
    const auto end = source.find("case PM4ItOpcode::IndexType:", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = std::string_view{source}.substr(begin, end - begin);

    // IT_SET_PREDICATION reads two body dwords. The handler must prove the declared packet is
    // present before forming a subspan, and this diagnostic-only path must never turn the guest
    // address into an unchecked host pointer while the operation semantics remain unimplemented.
    EXPECT_NE(body.find("count < 2"), std::string_view::npos);
    EXPECT_NE(body.find("dcb.size()"), std::string_view::npos);
    EXPECT_NE(body.find("dcb.subspan(1, count)"), std::string_view::npos);
    EXPECT_NE(body.find("DecodeLiverpoolSetPredication"), std::string_view::npos);
    EXPECT_EQ(body.find("reinterpret_cast<const volatile"), std::string_view::npos);
    EXPECT_EQ(body.find("IsValidMapping"), std::string_view::npos);
}

TEST(AscSubmissionLifetime, DeferredDmaPatchResolvesAgainstOwnedSnapshot) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto begin = source.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto end = source.find("void Liverpool::SubmitGfx(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    EXPECT_NE(body.find("guest_address - snapshot_base_addr"), std::string::npos);
    EXPECT_NE(body.find("snapshot_header"), std::string::npos);
    EXPECT_NE(body.find("indirect_patches.emplace_back(dst_addr - sizeof(PM4Header), src_addr)"),
              std::string::npos);
    EXPECT_EQ(body.find("indirect_patches.emplace_back(snapshot_header, src_addr)"),
              std::string::npos);
}

TEST(AscSubmissionLifetime, DeferredDmaPatchMatchesGuestAddressAcrossSnapshots) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto begin = source.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto end = source.find("void Liverpool::SubmitGfx(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    // DMA patches target the guest command stream, but ProcessCompute parses an immutable host
    // snapshot. Match both sides by guest address so copied and reassembled host buffers cannot
    // make a valid dispatch patch disappear.
    EXPECT_NE(body.find("VAddr header_address"), std::string::npos);
    EXPECT_NE(body.find("VAddr acb_guest_base_addr = source_base_addr"), std::string::npos);
    EXPECT_NE(body.find("VAddr packet_guest_base_addr = acb_guest_base_addr"), std::string::npos);
    EXPECT_NE(body.find("packet_guest_base_addr = queue.tmp_source_base_addr"),
              std::string::npos);
    EXPECT_NE(body.find("std::ranges::find(indirect_patches, packet_guest_base_addr"),
              std::string::npos);
    EXPECT_EQ(body.find("&IndirectPatch::header)"), std::string::npos);
}

TEST(AscSubmissionLifetime, SplitRingContinuationRefreshesSubmissionOwner) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto begin = source.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto end = source.find("void Liverpool::SubmitGfx(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    const auto continuation = body.find("if (queue.tmp_dwords > 0)");
    ASSERT_NE(continuation, std::string::npos);
    const auto owner_retention = body.find("retained_submission_owners.push_back(owner)", continuation);
    ASSERT_NE(owner_retention, std::string::npos);
    const auto owner_refresh = body.find("CopyWithIndirectPacket", continuation);
    ASSERT_NE(owner_refresh, std::string::npos);
    EXPECT_LT(owner_retention, owner_refresh);
    EXPECT_NE(body.find("queue.tmp_packet.data(), packet_words", owner_refresh),
              std::string::npos);
    EXPECT_NE(body.find("queue.tmp_source_base_addr = acb_guest_base_addr"),
              std::string::npos);
    EXPECT_NE(body.find("packet_guest_base_addr = queue.tmp_source_base_addr", continuation),
              std::string::npos);
}

TEST(AscSubmissionLifetime, DeferredDmaPatchRequiresFullPayloadInOwnedSnapshot) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto begin = source.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto end = source.find("void Liverpool::SubmitGfx(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    // A 12-byte dispatch-dimension copy whose destination is only four bytes from the snapshot
    // end must not be classified as a patch: the complete DMA payload must fit in the snapshot.
    EXPECT_NE(body.find("copy_bytes > snapshot_size - destination_offset"), std::string::npos);
    EXPECT_NE(body.find("snapshot_header_at_guest_address(packet_snapshot"),
              std::string::npos);
}

TEST(AscSubmissionLifetime, NestedCoroutineFramesAreDestroyedAfterCompletion) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};

    const auto process_compute_begin = source.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto process_compute_end = source.find("void Liverpool::SubmitGfx(", process_compute_begin);
    ASSERT_NE(process_compute_begin, std::string::npos);
    ASSERT_NE(process_compute_end, std::string::npos);
    const auto process_compute = source.substr(process_compute_begin,
                                               process_compute_end - process_compute_begin);
    const auto asc_child_begin = process_compute.find("case PM4ItOpcode::IndirectBuffer:");
    const auto asc_child_end = process_compute.find("case PM4ItOpcode::DmaData:", asc_child_begin);
    ASSERT_NE(asc_child_begin, std::string::npos);
    ASSERT_NE(asc_child_end, std::string::npos);
    const auto asc_child = process_compute.substr(asc_child_begin, asc_child_end - asc_child_begin);
    EXPECT_NE(asc_child.find("while (!task.handle.done()"), std::string::npos);
    EXPECT_NE(asc_child.find("task.handle.destroy()"), std::string::npos);

    const auto process_ce_begin = source.find("Liverpool::Task Liverpool::ProcessCeUpdate(");
    const auto process_ce_end = source.find("Liverpool::Task Liverpool::ProcessGraphics(", process_ce_begin);
    ASSERT_NE(process_ce_begin, std::string::npos);
    ASSERT_NE(process_ce_end, std::string::npos);
    const auto process_ce = source.substr(process_ce_begin, process_ce_end - process_ce_begin);
    const auto ce_child_begin = process_ce.find("case PM4ItOpcode::IndirectBufferConst:");
    const auto ce_child_end = process_ce.find("default:", ce_child_begin);
    ASSERT_NE(ce_child_begin, std::string::npos);
    ASSERT_NE(ce_child_end, std::string::npos);
    const auto ce_child = process_ce.substr(ce_child_begin, ce_child_end - ce_child_begin);
    EXPECT_NE(ce_child.find("while (!task.handle.done()"), std::string::npos);
    EXPECT_NE(ce_child.find("task.handle.destroy()"), std::string::npos);
}

TEST(AscSubmissionLifetime, InactiveQueueStopsEveryDeferredAscWaitLoop) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto begin = source.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto end = source.find("void Liverpool::SubmitGfx(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    const auto branch = [&](std::string_view marker, std::string_view next_marker) {
        const auto branch_begin = body.find(marker);
        const auto branch_end = body.find(next_marker, branch_begin);
        EXPECT_NE(branch_begin, std::string::npos);
        EXPECT_NE(branch_end, std::string::npos);
        return body.substr(branch_begin, branch_end - branch_begin);
    };

    const auto indirect = branch("case PM4ItOpcode::IndirectBuffer:",
                                 "case PM4ItOpcode::DmaData:");
    const auto rewind = branch("case PM4ItOpcode::Rewind:", "case PM4ItOpcode::SetShReg:");
    const auto semaphore = branch("case PM4ItOpcode::MemSemaphore:",
                                  "case PM4ItOpcode::WaitRegMem:");
    const auto wait_reg_mem = branch("case PM4ItOpcode::WaitRegMem:",
                                     "case PM4ItOpcode::ReleaseMem:");

    for (const auto* wait_branch : {&indirect, &rewind, &semaphore, &wait_reg_mem}) {
        EXPECT_NE(wait_branch->find("queue_state->active.load(std::memory_order_acquire)"),
                  std::string::npos);
    }
}

TEST(AscSubmissionLifetime, PartialPacketContinuationIsBoundedByScratchCapacity) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto begin = source.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto end = source.find("void Liverpool::SubmitGfx(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    const auto split_begin = body.find("// If the packet is split across ring boundary");
    const auto split_end = body.find("if (header->type == 2)", split_begin);
    ASSERT_NE(split_begin, std::string::npos);
    ASSERT_NE(split_end, std::string::npos);
    const auto split = body.substr(split_begin, split_end - split_begin);
    EXPECT_NE(split.find("queue.tmp_packet.size()"), std::string::npos);
    EXPECT_NE(split.find("queue_state->active.store(false"), std::string::npos);
    EXPECT_NE(split.find("next_dw_off > queue.tmp_packet.size()"), std::string::npos);
    EXPECT_NE(split.find("acb.size() > queue.tmp_packet.size()"), std::string::npos);
    EXPECT_NE(split.find("acb.size_bytes()"), std::string::npos);

    const auto continuation_begin = body.find("if (queue.tmp_dwords > 0)");
    const auto continuation_end = body.find("// If the packet is split across ring boundary",
                                             continuation_begin);
    ASSERT_NE(continuation_begin, std::string::npos);
    ASSERT_NE(continuation_end, std::string::npos);
    const auto continuation = body.substr(continuation_begin, continuation_end - continuation_begin);
    EXPECT_NE(continuation.find("packet_words < queue.tmp_dwords"), std::string::npos);
    EXPECT_NE(continuation.find("packet_words - queue.tmp_dwords > acb.size()"),
              std::string::npos);
    EXPECT_NE(continuation.find(
                  "packet_words - queue.tmp_dwords > queue.tmp_packet.size() - queue.tmp_dwords"),
              std::string::npos);
}

TEST(AscSubmissionLifetime, ReadPointerAdvanceUsesWideIntermediate) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto begin = source.find("void Liverpool::AdvanceAscReadPointer(");
    const auto end = source.find("void Liverpool::SubmitGfx(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_NE(body.find("static_cast<u64>(queue_state->read_offset_dw)"), std::string::npos);
    EXPECT_NE(body.find("+ delta) %"), std::string::npos);
    EXPECT_NE(body.find("queue_state->ring_size_dw"), std::string::npos);
}

TEST(AscSubmissionLifetime, QueueMappingResetsAndBoundsItsSubmissionOffset) {
    std::ifstream input{SHADPS4_GNM_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto map_begin = source.find("int PS4_SYSV_ABI sceGnmMapComputeQueue(");
    const auto map_end = source.find("int PS4_SYSV_ABI sceGnmMapComputeQueueWithPriority(", map_begin);
    const auto ding_begin = source.find("void PS4_SYSV_ABI sceGnmDingDong(");
    const auto ding_end = source.find("void PS4_SYSV_ABI sceGnmDingDongForWorkload(", ding_begin);
    ASSERT_NE(map_begin, std::string::npos);
    ASSERT_NE(map_end, std::string::npos);
    ASSERT_NE(ding_begin, std::string::npos);
    ASSERT_NE(ding_end, std::string::npos);
    const auto map = source.substr(map_begin, map_end - map_begin);
    const auto ding = source.substr(ding_begin, ding_end - ding_begin);
    EXPECT_NE(map.find("next_offs_dw = 0"), std::string::npos);
    EXPECT_NE(ding.find("next_offs_dw > asc_queue->ring_size_dw"), std::string::npos);
}

TEST(AscSubmissionLifetime, DeferredWorkerRetainsSnapshotAndQueueUntilItRuns) {
    std::array<u32, 4> ring{0x10, 0x20, 0x30, 0x40};
    auto submission = AmdGpu::PrepareComputeSubmission(ring);

    struct QueueState {
        std::atomic_bool active{true};
    };
    auto queue = std::make_shared<QueueState>();
    const std::weak_ptr<QueueState> retired_queue = queue;

    std::promise<void> release_worker;
    auto worker_release = release_worker.get_future().share();
    auto worker = std::async(std::launch::async,
                             [owner = submission.owner, acb = submission.acb, queue,
                              worker_release] {
                                 worker_release.wait();
                                 return std::pair{
                                     !queue->active.load(std::memory_order_acquire),
                                     std::ranges::equal(acb, std::array<u32, 4>{0x10, 0x20, 0x30,
                                                                                0x40}),
                                 };
                             });

    // Model the producer returning from DingDong: its ring storage and queue slot can be
    // mutated/retired immediately, while the deferred task owns immutable copies and state.
    submission.owner.reset();
    submission.acb = {};
    ring.fill(0);
    queue->active.store(false, std::memory_order_release);
    queue.reset();
    EXPECT_FALSE(retired_queue.expired());

    release_worker.set_value();
    const auto [observed_retirement, preserved_commands] = worker.get();
    EXPECT_TRUE(observed_retirement);
    EXPECT_TRUE(preserved_commands);
    EXPECT_TRUE(retired_queue.expired());
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

TEST(AscSubmissionLifetime, DeferredComputeUsesQueueOwnedComputeRegisters) {
    const auto liverpool_path = std::filesystem::path{LIVERPOOL_SOURCE_PATH};
    std::ifstream header_input{liverpool_path.parent_path() / "liverpool.h", std::ios::binary};
    ASSERT_TRUE(header_input);
    const std::string header{std::istreambuf_iterator<char>{header_input}, {}};
    const auto queue_begin = header.find("struct AscQueueInfo");
    const auto queue_end = header.find("void SubmitAsc", queue_begin);
    ASSERT_NE(queue_begin, std::string::npos);
    ASSERT_NE(queue_end, std::string::npos);
    const auto queue = header.substr(queue_begin, queue_end - queue_begin);
    EXPECT_NE(queue.find("ComputeProgram cs_state"), std::string::npos);

    std::ifstream source_input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(source_input);
    const std::string source{std::istreambuf_iterator<char>{source_input}, {}};
    const auto process_begin = source.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto process_end = source.find("void Liverpool::SubmitGfx(", process_begin);
    ASSERT_NE(process_begin, std::string::npos);
    ASSERT_NE(process_end, std::string::npos);
    const auto process = source.substr(process_begin, process_end - process_begin);
    EXPECT_NE(process.find("auto& cs_program = queue.cs_state"), std::string::npos);
    EXPECT_EQ(process.find("GetCsRegs()"), std::string::npos);
    EXPECT_NE(process.find("publish_compute_state_for_dispatch"), std::string::npos);
    EXPECT_NE(process.find("mapped_queues[vqid + 1].cs_state = queue.cs_state"),
              std::string::npos);
}

TEST(AscSubmissionLifetime, DeferredAscRegisterAccessCannotMutateGraphicsImage) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto begin = source.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto end = source.find("void Liverpool::SubmitGfx(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto process = source.substr(begin, end - begin);

    EXPECT_NE(process.find("queue.TryReadComputeRegister(wait.Reg())"), std::string::npos);
    EXPECT_EQ(process.find("value = regs.reg_array[reg]"), std::string::npos);
    EXPECT_NE(process.find("Ignoring ASC SetShReg targeting non-compute register"),
              std::string::npos);
    EXPECT_EQ(process.find("regs.reg_array[reg_index]"), std::string::npos);
}

TEST(AscSubmissionLifetime, SetShRegChecksPacketCountAndComputeStateBounds) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto process_begin = source.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto process_end = source.find("void Liverpool::SubmitGfx(", process_begin);
    ASSERT_NE(process_begin, std::string::npos);
    ASSERT_NE(process_end, std::string::npos);
    const auto process = source.substr(process_begin, process_end - process_begin);
    const auto set_begin = process.find("case PM4ItOpcode::SetShReg:");
    const auto set_end = process.find("case PM4ItOpcode::SetQueueReg:", set_begin);
    ASSERT_NE(set_begin, std::string::npos);
    ASSERT_NE(set_end, std::string::npos);
    const auto set = process.substr(set_begin, set_end - set_begin);
    EXPECT_NE(set.find("packet_words == 0"), std::string::npos);
    EXPECT_NE(set.find("set_size > sizeof(ComputeProgram) -"), std::string::npos);
    EXPECT_NE(set.find("reg_offset <"), std::string::npos);
}

TEST(AscSubmissionLifetime, ProcessComputeRechecksQueueActivityAfterCommandProcessing) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto process_begin = source.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto process_end = source.find("void Liverpool::SubmitGfx(", process_begin);
    ASSERT_NE(process_begin, std::string::npos);
    ASSERT_NE(process_end, std::string::npos);
    const auto process = source.substr(process_begin, process_end - process_begin);
    const auto commands = process.find("ProcessCommands();");
    ASSERT_NE(commands, std::string::npos);
    const auto header = process.find("auto* header =", commands);
    ASSERT_NE(header, std::string::npos);
    const auto between = process.substr(commands, header - commands);
    EXPECT_NE(between.find("queue_state->active.load(std::memory_order_acquire)"),
              std::string::npos);
}

TEST(AscSubmissionLifetime, MappedQueueCountPublicationIsAtomic) {
    const auto liverpool_path = std::filesystem::path{LIVERPOOL_SOURCE_PATH};
    std::ifstream header_input{liverpool_path.parent_path() / "liverpool.h", std::ios::binary};
    ASSERT_TRUE(header_input);
    const std::string header{std::istreambuf_iterator<char>{header_input}, {}};
    EXPECT_NE(header.find("std::atomic<u32> num_mapped_queues"), std::string::npos);

    std::ifstream source_input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(source_input);
    const std::string source{std::istreambuf_iterator<char>{source_input}, {}};
    const auto process_begin = source.find("void Liverpool::Process(std::stop_token stoken)");
    const auto process_end = source.find("Liverpool::Task Liverpool::ProcessCeUpdate(",
                                        process_begin);
    const auto submit_begin = source.find("void Liverpool::SubmitAsc(");
    const auto submit_end = source.find("void Liverpool::SubmitDone(", submit_begin);
    ASSERT_NE(process_begin, std::string::npos);
    ASSERT_NE(process_end, std::string::npos);
    ASSERT_NE(submit_begin, std::string::npos);
    ASSERT_NE(submit_end, std::string::npos);

    const auto process = source.substr(process_begin, process_end - process_begin);
    const auto submit = source.substr(submit_begin, submit_end - submit_begin);
    EXPECT_NE(process.find("num_mapped_queues.load(std::memory_order_acquire)"),
              std::string::npos);
    EXPECT_NE(submit.find("num_mapped_queues.load(std::memory_order_relaxed)"),
              std::string::npos);
    EXPECT_NE(submit.find("num_mapped_queues.store("), std::string::npos);
}

TEST(AscSubmissionLifetime, QueueMappingRejectsFailedReadPointerInitialization) {
    std::ifstream input{SHADPS4_GNM_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto map_begin = source.find("int PS4_SYSV_ABI sceGnmMapComputeQueue(");
    const auto map_end = source.find("int PS4_SYSV_ABI sceGnmMapComputeQueueWithPriority(",
                                     map_begin);
    ASSERT_NE(map_begin, std::string::npos);
    ASSERT_NE(map_end, std::string::npos);
    const auto map = source.substr(map_begin, map_end - map_begin);

    EXPECT_NE(map.find("read_ptr_initialized"), std::string::npos);
    EXPECT_NE(map.find("if (!read_ptr_initialized)"), std::string::npos);
    EXPECT_NE(map.find("liverpool->asc_queues.erase(vqid)"), std::string::npos);
    EXPECT_NE(map.find("ORBIS_GNM_ERROR_COMPUTEQUEUE_INVALID_READ_PTR_ADDR"),
              std::string::npos);
}

TEST(AscSubmissionLifetime, GraphicsParserRejectsTruncatedType3BeforeBodyAccess) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto begin = source.find("Liverpool::Task Liverpool::ProcessGraphics(");
    const auto end = source.find("template <bool is_indirect>", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    // A deferred graphics submission can end at any dword. Validate the complete Type-3
    // packet before switching on an opcode or reading its payload fields.
    EXPECT_NE(body.find("packet_words > dcb.size()"), std::string::npos);
    EXPECT_NE(body.find("Ignoring truncated graphics PM4 packet"), std::string::npos);
}

TEST(AscSubmissionLifetime, GraphicsSetShRegChecksPacketCountAndComputeStateBounds) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto process_begin = source.find("Liverpool::Task Liverpool::ProcessGraphics(");
    const auto process_end = source.find("template <bool is_indirect>", process_begin);
    ASSERT_NE(process_begin, std::string::npos);
    ASSERT_NE(process_end, std::string::npos);
    const auto process = source.substr(process_begin, process_end - process_begin);
    const auto set_begin = process.find("case PM4ItOpcode::SetShReg:");
    const auto set_end = process.find("case PM4ItOpcode::SetUconfigReg:", set_begin);
    ASSERT_NE(set_begin, std::string::npos);
    ASSERT_NE(set_end, std::string::npos);
    const auto set = process.substr(set_begin, set_end - set_begin);

    // Graphics submissions can be truncated or target the final compute-state register. Keep the
    // packet and the state copy bounded before touching either payload or state storage.
    EXPECT_NE(set.find("packet_words == 0"), std::string::npos);
    EXPECT_NE(set.find("reg_offset < compute_reg_base + compute_state_words"),
              std::string::npos);
    EXPECT_NE(set.find("set_size > sizeof(ComputeProgram) - state_offset"), std::string::npos);
}

TEST(AscSubmissionLifetime, ComputeParserHandlesCondExecWithCheckedConditionAndBoundedSkip) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto process_begin = source.find("Liverpool::Task Liverpool::ProcessCompute(");
    const auto process_end = source.find("void Liverpool::SubmitGfx(", process_begin);
    ASSERT_NE(process_begin, std::string::npos);
    ASSERT_NE(process_end, std::string::npos);
    const auto process = source.substr(process_begin, process_end - process_begin);
    const auto cond_begin = process.find("case PM4ItOpcode::CondExec:");
    const auto cond_end = process.find("default:", cond_begin);
    ASSERT_NE(cond_begin, std::string::npos);
    ASSERT_NE(cond_end, std::string::npos);
    const auto cond_exec = process.substr(cond_begin, cond_end - cond_begin);

    // A valid ASC conditional block must not fall into UNREACHABLE. The condition is a guest
    // address and the skip count must be bounded by the immutable submission snapshot before the
    // parser advances its ring/read-pointer accounting.
    EXPECT_NE(cond_exec.find("try_read_guest"), std::string::npos);
    EXPECT_NE(cond_exec.find("exec_count"), std::string::npos);
    EXPECT_NE(cond_exec.find("acb.size()"), std::string::npos);
    EXPECT_NE(cond_exec.find("AdvanceAscReadPointer"), std::string::npos);
}

TEST(AscSubmissionLifetime, ConstantEngineValidatesPacketAndConstantRamBounds) {
    std::ifstream input{LIVERPOOL_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto begin = source.find("Liverpool::Task Liverpool::ProcessCeUpdate(");
    const auto end = source.find("Liverpool::Task Liverpool::ProcessGraphics(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto process_ce = source.substr(begin, end - begin);

    // CE executes later on the GPU worker from an immutable CCB snapshot. The header and every
    // field/body access must be bounded before opcode-specific code can read a truncated packet.
    EXPECT_NE(process_ce.find("header->type3.NumWords()"), std::string::npos);
    EXPECT_NE(process_ce.find("static_cast<size_t>(header->type3.NumWords()) + 1"),
              std::string::npos);
    EXPECT_EQ(process_ce.find("header->type3.count.Value()"), std::string::npos);
    EXPECT_NE(process_ce.find("packet_words > ccb.size()"), std::string::npos);
    EXPECT_NE(process_ce.find("Ignoring truncated CE PM4 packet"), std::string::npos);

    // A malformed WRITE_CONST_RAM must not write past the fixed 48-KiB CE heap even when its
    // packet itself is fully present in the snapshot.
    const auto write_begin = process_ce.find("case PM4ItOpcode::WriteConstRam:");
    const auto write_end = process_ce.find("case PM4ItOpcode::DumpConstRam:", write_begin);
    ASSERT_NE(write_begin, std::string::npos);
    ASSERT_NE(write_end, std::string::npos);
    const auto write = process_ce.substr(write_begin, write_end - write_begin);
    EXPECT_NE(write.find("offset > cblock.constants_heap.size()"), std::string::npos);
    EXPECT_NE(write.find("size > cblock.constants_heap.size() - offset"), std::string::npos);
    EXPECT_NE(write.find("data() + offset"), std::string::npos);
}

} // namespace
} // namespace Libraries::GnmDriver
