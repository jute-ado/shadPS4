// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>
#include <atomic>
#include <chrono>
#include <future>

#include <gtest/gtest.h>

#include "core/memory_range_identity.h"
#include "core/memory_retirement_provider.h"
#include "core/optimistic_memory_retirement.h"
#include "core/physical_release_plan.h"
#include "core/libraries/gnmdriver/submission_retirement.h"
#include "video_core/amdgpu/submission_completion_queue.h"

namespace Core {

struct MemoryRangeIdentityInspector {
    static size_t BoundaryCount(const MemoryRangeIdentity& identities) {
        return identities.boundaries.size();
    }
};

namespace {

TEST(MemoryRangeIdentity, RebindingIdenticalMetadataStillChangesIdentity) {
    MemoryRangeIdentity identities;
    ASSERT_TRUE(identities.Stamp(0x10000, 0x4000));
    const auto old = identities.Capture(0x10000, 0x4000);
    ASSERT_TRUE(old);
    EXPECT_TRUE(identities.Matches(*old));
    ASSERT_TRUE(identities.Stamp(0x10000, 0x4000));
    EXPECT_FALSE(identities.Matches(*old));
}

TEST(MemoryRangeIdentity, UnrelatedAllocationDoesNotInvalidateRetirement) {
    MemoryRangeIdentity identities;
    ASSERT_TRUE(identities.Stamp(0x10000, 0x4000));
    const auto old = identities.Capture(0x10000, 0x4000);
    ASSERT_TRUE(old);
    ASSERT_TRUE(identities.Stamp(0x20000, 0x4000));
    EXPECT_TRUE(identities.Matches(*old));
}

TEST(MemoryRangeIdentity, PartialOverlapPreservesUnchangedSides) {
    MemoryRangeIdentity identities;
    ASSERT_TRUE(identities.Stamp(0x10000, 0xc000));
    const auto left = identities.Capture(0x10000, 0x4000);
    const auto middle = identities.Capture(0x14000, 0x4000);
    const auto right = identities.Capture(0x18000, 0x4000);
    ASSERT_TRUE(left && middle && right);
    ASSERT_TRUE(identities.Stamp(0x14000, 0x4000));
    EXPECT_TRUE(identities.Matches(*left));
    EXPECT_FALSE(identities.Matches(*middle));
    EXPECT_TRUE(identities.Matches(*right));
}

TEST(MemoryRangeIdentity, PhysicalAllocationReuseAndAliasChangeInvalidateTheirRange) {
    MemoryRangeIdentity physical;
    ASSERT_TRUE(physical.Stamp(0x4000, 0x8000));
    const auto allocated = physical.Capture(0x4000, 0x8000);
    ASSERT_TRUE(allocated);
    // Adding an alias or recycling this physical allocation stamps the affected physical range.
    ASSERT_TRUE(physical.Stamp(0x8000, 0x4000));
    EXPECT_FALSE(physical.Matches(*allocated));
}

TEST(MemoryRangeIdentity, SnapshotRepresentsHolesWithoutInventingBacking) {
    MemoryRangeIdentity identities;
    ASSERT_TRUE(identities.Stamp(0x14000, 0x4000));
    const auto snapshot = identities.Capture(0x10000, 0xc000);
    ASSERT_TRUE(snapshot);
    ASSERT_EQ(snapshot->parts.size(), 3u);
    EXPECT_EQ(snapshot->parts[0], (MemoryRangeIdentity::Part{0x10000, 0x4000, 0}));
    EXPECT_NE(snapshot->parts[1].identity, 0u);
    EXPECT_EQ(snapshot->parts[2], (MemoryRangeIdentity::Part{0x18000, 0x4000, 0}));
}

TEST(MemoryRangeIdentity, StructuralVmaChangesNeedNotStampBackingIdentity) {
    MemoryRangeIdentity identities;
    ASSERT_TRUE(identities.Stamp(0x10000, 0xc000));
    const auto before = identities.Capture(0x10000, 0xc000);
    // A guest VMA split/merge/name edit does not mutate this independent backing index.
    const auto after = identities.Capture(0x10000, 0xc000);
    ASSERT_TRUE(before && after);
    EXPECT_EQ(*before, *after);
}

TEST(MemoryRangeIdentity, OverflowFailsBeforeChangingExistingRanges) {
    MemoryRangeIdentity identities;
    ASSERT_TRUE(identities.Stamp(0x10000, 0x4000));
    const auto old = identities.Capture(0x10000, 0x4000);
    ASSERT_TRUE(old);
    const auto max = std::numeric_limits<std::uint64_t>::max();
    EXPECT_FALSE(identities.Stamp(max - 3, 8));
    EXPECT_FALSE(identities.Capture(max - 3, 8));
    EXPECT_TRUE(identities.Matches(*old));
}

TEST(MemoryRangeIdentity, EmptyRangeIsAReadOnlyNoop) {
    MemoryRangeIdentity identities;
    ASSERT_TRUE(identities.Stamp(0x10000, 0x4000));
    const auto old = identities.Capture(0x10000, 0x4000);
    ASSERT_TRUE(old);
    EXPECT_TRUE(identities.Stamp(0x10000, 0));
    EXPECT_TRUE(identities.Matches(*old));
    const auto empty = identities.Capture(0x10000, 0);
    ASSERT_TRUE(empty);
    EXPECT_TRUE(empty->parts.empty());
}

TEST(MemoryRangeIdentity, ReplacingAcrossSeveralOldBoundariesLeavesNoStaleInterior) {
    MemoryRangeIdentity identities;
    ASSERT_TRUE(identities.Stamp(4, 4));
    ASSERT_TRUE(identities.Stamp(12, 4));
    ASSERT_TRUE(identities.Stamp(0, 20));
    const auto whole = identities.Capture(0, 20);
    ASSERT_TRUE(whole);
    ASSERT_EQ(whole->parts.size(), 1u);
    EXPECT_NE(whole->parts.front().identity, 0u);
    const auto outside = identities.Capture(20, 4);
    ASSERT_TRUE(outside);
    EXPECT_EQ(outside->parts, (std::vector<MemoryRangeIdentity::Part>{{20, 4, 0}}));
}

TEST(MemoryRangeIdentity, TouchingRangesDoNotInvalidateEachOther) {
    MemoryRangeIdentity identities;
    ASSERT_TRUE(identities.Stamp(4, 4));
    const auto middle = identities.Capture(4, 4);
    ASSERT_TRUE(middle);
    ASSERT_TRUE(identities.Stamp(0, 4));
    ASSERT_TRUE(identities.Stamp(8, 4));
    EXPECT_TRUE(identities.Matches(*middle));
}

TEST(MemoryRangeIdentity, HighestRepresentableEndIsHandledWithoutWrapping) {
    MemoryRangeIdentity identities;
    const auto max = std::numeric_limits<std::uint64_t>::max();
    ASSERT_TRUE(identities.Stamp(max - 8, 8));
    const auto last = identities.Capture(max - 8, 8);
    ASSERT_TRUE(last);
    ASSERT_EQ(last->parts.size(), 1u);
    EXPECT_EQ(last->parts.front().size, 8u);
    EXPECT_TRUE(identities.Matches(*last));
    ASSERT_TRUE(identities.Stamp(max - 4, 4));
    EXPECT_FALSE(identities.Matches(*last));
}

TEST(MemoryRangeIdentity, FreePrunesMetadataButDoesNotPermitLiveIdentityAba) {
    MemoryRangeIdentity identities;
    ASSERT_TRUE(identities.Stamp(0x10000, 0x4000));
    const auto live = identities.Capture(0x10000, 0x4000);
    ASSERT_TRUE(live);
    ASSERT_TRUE(identities.Erase(0x10000, 0x4000));
    EXPECT_FALSE(identities.Matches(*live));
    EXPECT_EQ(MemoryRangeIdentityInspector::BoundaryCount(identities), 0u);
    ASSERT_TRUE(identities.Stamp(0x10000, 0x4000));
    EXPECT_FALSE(identities.Matches(*live));
}

TEST(MemoryRangeIdentity, RepeatedFreedAddressesDoNotAccumulatePrivateBoundaries) {
    MemoryRangeIdentity identities;
    for (std::uint64_t address = 0; address < 4096; address += 4) {
        ASSERT_TRUE(identities.Stamp(address, 4));
        ASSERT_TRUE(identities.Erase(address, 4));
    }
    EXPECT_EQ(MemoryRangeIdentityInspector::BoundaryCount(identities), 0u);
}

TEST(MemoryRangeIdentity, PartialFreeCoalescesHolesWithoutChangingLiveSides) {
    MemoryRangeIdentity identities;
    ASSERT_TRUE(identities.Stamp(0, 20));
    const auto left = identities.Capture(0, 4);
    const auto right = identities.Capture(16, 4);
    ASSERT_TRUE(left && right);
    ASSERT_TRUE(identities.Erase(4, 4));
    ASSERT_TRUE(identities.Erase(8, 8));
    EXPECT_TRUE(identities.Matches(*left));
    EXPECT_TRUE(identities.Matches(*right));
    const auto whole = identities.Capture(0, 20);
    ASSERT_TRUE(whole);
    ASSERT_EQ(whole->parts.size(), 3u);
    EXPECT_EQ(whole->parts[1], (MemoryRangeIdentity::Part{4, 12, 0}));
    EXPECT_EQ(MemoryRangeIdentityInspector::BoundaryCount(identities), 4u);
}

TEST(MemoryRangeIdentity, FreeHoleOwnsNoLiveIdentityAcrossTransientMapping) {
    MemoryRangeIdentity identities;
    const auto hole = identities.Capture(0, 16);
    ASSERT_TRUE(hole);
    ASSERT_TRUE(identities.Stamp(0, 16));
    EXPECT_FALSE(identities.Matches(*hole));
    ASSERT_TRUE(identities.Erase(0, 16));
    EXPECT_TRUE(identities.Matches(*hole));
    // CP-issued sequence validation, not a hole's identity, covers work submitted in between.
}

TEST(MemoryRangeIdentity, InvalidEraseDoesNotChangeLiveIdentity) {
    MemoryRangeIdentity identities;
    ASSERT_TRUE(identities.Stamp(0, 16));
    const auto live = identities.Capture(0, 16);
    ASSERT_TRUE(live);
    EXPECT_FALSE(identities.Erase(std::numeric_limits<std::uint64_t>::max() - 3, 8));
    EXPECT_TRUE(identities.Matches(*live));
}

using namespace std::chrono_literals;

TEST(SubmissionRetirement, TicketWaitsForItsExactIssuedWorkWithoutSubmitDone) {
    AmdGpu::SubmissionCompletionQueue queue;
    const auto sequence = queue.IssueSubmission();
    const auto ticket = queue.CaptureCompletionTicket();
    EXPECT_EQ(ticket.sequence, sequence);
    auto waiter = std::async(std::launch::async, [&] { return ticket.Wait(); });
    const auto before_completion = waiter.wait_for(20ms);
    queue.CompleteSubmission(sequence);
    EXPECT_EQ(before_completion, std::future_status::timeout);
    ASSERT_EQ(waiter.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(waiter.get());
}

TEST(SubmissionRetirement, IdleTicketCompletesOnTheOwnerDrain) {
    AmdGpu::SubmissionCompletionQueue queue;
    const auto ticket = queue.CaptureCompletionTicket();
    auto waiter = std::async(std::launch::async, [&] { return ticket.Wait(); });
    const auto before_drain = waiter.wait_for(20ms);
    queue.DrainReadyBoundaries();
    EXPECT_EQ(before_drain, std::future_status::timeout);
    ASSERT_EQ(waiter.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(waiter.get());
}

TEST(SubmissionRetirement, CancelledTicketWakesWithoutInventingCompletion) {
    AmdGpu::SubmissionCompletionQueue queue;
    const auto sequence = queue.IssueSubmission();
    const auto ticket = queue.CaptureCompletionTicket();
    std::stop_source stop;
    auto waiter = std::async(std::launch::async, [&] { return ticket.Wait(stop.get_token()); });
    const auto before_cancel = waiter.wait_for(20ms);
    stop.request_stop();
    EXPECT_EQ(before_cancel, std::future_status::timeout);
    ASSERT_EQ(waiter.wait_for(1s), std::future_status::ready);
    EXPECT_FALSE(waiter.get());
    queue.CompleteSubmission(sequence); // A late normal completion does not revive cancellation.
    EXPECT_FALSE(ticket.Wait(stop.get_token()));
}

TEST(SubmissionRetirement, PriorityFinishCannotRetireUnparsedSubmission) {
    Libraries::GnmDriver::SubmissionGate gate;
    AmdGpu::SubmissionCompletionQueue queue;
    const auto sequence = queue.IssueSubmission();
    std::atomic<bool> backing_live{true};
    std::atomic<unsigned> finished{};
    auto retire = std::async(std::launch::async, [&] {
        auto admission = Libraries::GnmDriver::AcquireSubmissionRetirement(
            gate, [&] {
                auto ticket = queue.CaptureCompletionTicket();
                queue.DrainReadyBoundaries();
                return ticket;
            },
            [&] { return queue.IssuedSequence(); }, [&] { ++finished; });
        if (admission) {
            backing_live = false;
        }
        return bool(admission);
    });
    const auto before_cp = retire.wait_for(20ms);
    const auto backing_before_cp = backing_live.load();
    const auto finishes_before_cp = finished.load();
    queue.CompleteSubmission(sequence);
    // The idle retry case is also safe if the test host was slow to start the worker.
    queue.DrainReadyBoundaries();
    EXPECT_EQ(before_cp, std::future_status::timeout);
    EXPECT_TRUE(backing_before_cp);
    EXPECT_EQ(finishes_before_cp, 0u);
    ASSERT_EQ(retire.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(retire.get());
    EXPECT_EQ(finished.load(), 1u);
    EXPECT_FALSE(backing_live.load());
}

TEST(MemoryRetirementProvider, ClosedProviderNeverCallsRetiredOwner) {
    unsigned calls{};
    MemoryRetirementProvider provider{[&](std::stop_token) {
        ++calls;
        return std::optional<MemoryRetirementProvider::Token>{std::in_place};
    }};
    provider.CloseAndWait();
    EXPECT_FALSE(provider.Lease());
    EXPECT_FALSE(provider.Retire());
    EXPECT_EQ(calls, 0u);
}

TEST(MemoryRetirementProvider, FreshLeaseDoesNotRequestSubmissionRetirement) {
    unsigned calls{};
    MemoryRetirementProvider provider{[&](std::stop_token) {
        ++calls;
        return std::optional<MemoryRetirementProvider::Token>{std::in_place};
    }};
    EXPECT_TRUE(provider.Lease());
    EXPECT_EQ(calls, 0u);
}

TEST(MemoryRetirementProvider, CloseWaitsForFreshMappingLifetime) {
    MemoryRetirementProvider provider{[](std::stop_token) {
        return std::optional<MemoryRetirementProvider::Token>{std::in_place};
    }};
    auto lease = provider.Lease();
    auto close = std::async(std::launch::async, [&] { provider.CloseAndWait(); });
    const auto before_release = close.wait_for(20ms);
    lease.reset();
    EXPECT_EQ(before_release, std::future_status::timeout);
    ASSERT_EQ(close.wait_for(1s), std::future_status::ready);
    close.get();
}

TEST(MemoryRetirementProvider, OwnerCancellationPropagatesWithoutMutation) {
    MemoryRetirementProvider provider{[](std::stop_token) {
        return std::optional<MemoryRetirementProvider::Token>{};
    }};
    EXPECT_FALSE(provider.Retire());
}

TEST(MemoryRetirementProvider, SubmissionGuardOutlivesTheRetireCall) {
    auto owner_guard = std::make_shared<int>(7);
    std::weak_ptr<int> weak_guard = owner_guard;
    MemoryRetirementProvider provider{[&](std::stop_token) {
        return std::optional<MemoryRetirementProvider::Token>{
            [guard = std::exchange(owner_guard, {})] {}};
    }};
    auto admission = provider.Retire();
    EXPECT_FALSE(weak_guard.expired());
    admission.reset();
    EXPECT_TRUE(weak_guard.expired());
}

TEST(MemoryRetirementProvider, FatalAcquisitionErrorsAreNotCancellation) {
    MemoryRetirementProvider provider{[](std::stop_token)
                                          -> std::optional<MemoryRetirementProvider::Token> {
        throw std::runtime_error("fatal GPU failure");
    }};
    EXPECT_THROW((void)provider.Retire(), std::runtime_error);
    provider.CloseAndWait(); // An exception must not leak an in-progress owner lease.
}

TEST(MemoryRetirementProvider, CloseCancelsPendingTicketAndLateOwnerCompletionIsSafe) {
    AmdGpu::SubmissionCompletionQueue queue;
    const auto sequence = queue.IssueSubmission();
    std::promise<void> entered;
    MemoryRetirementProvider provider{[&](std::stop_token stop) {
        auto ticket = queue.CaptureCompletionTicket();
        entered.set_value();
        if (!ticket.Wait(stop)) {
            return std::optional<MemoryRetirementProvider::Token>{};
        }
        return std::optional<MemoryRetirementProvider::Token>{std::in_place};
    }};
    auto retire = std::async(std::launch::async, [&] { return bool(provider.Retire()); });
    entered.get_future().wait();
    provider.CloseAndWait();
    EXPECT_FALSE(retire.get());
    queue.CompleteSubmission(sequence);
    EXPECT_FALSE(provider.Retire());
}

TEST(SubmissionRetirement, NewSubmissionDuringTicketWaitMustAlsoRetire) {
    Libraries::GnmDriver::SubmissionGate gate;
    AmdGpu::SubmissionCompletionQueue queue;
    const auto first = queue.IssueSubmission();
    std::promise<void> captured_first;
    std::promise<void> captured_second;
    unsigned capture_count{};
    std::atomic<unsigned> finishes{};
    auto retire = std::async(std::launch::async, [&] {
        return bool(Libraries::GnmDriver::AcquireSubmissionRetirement(
            gate, [&] {
                auto ticket = queue.CaptureCompletionTicket();
                (++capture_count == 1 ? captured_first : captured_second).set_value();
                return ticket;
            }, [&] { return queue.IssuedSequence(); }, [&] { ++finishes; }));
    });
    captured_first.get_future().wait();
    // This issuance must stay possible while an old WAIT_REG_MEM awaits new guest work.
    auto issue = gate.Enter();
    const auto second = queue.IssueSubmission();
    issue.unlock();
    queue.CompleteSubmission(first);
    captured_second.get_future().wait();
    EXPECT_EQ(finishes.load(), 0u);
    queue.CompleteSubmission(second);
    EXPECT_TRUE(retire.get());
    EXPECT_EQ(finishes.load(), 1u);
}

TEST(SubmissionRetirement, SameSequenceSubmitDoneMustFinishBeforeFinalAdmission) {
    Libraries::GnmDriver::SubmissionGate gate;
    AmdGpu::SubmissionCompletionQueue queue;
    const auto sequence = queue.IssueSubmission();
    std::promise<void> captured_first;
    std::promise<void> captured_retry;
    unsigned captures{};
    std::atomic<bool> submit_done_complete{};
    std::atomic<bool> finish_after_submit_done{};
    std::stop_source stop;
    auto retire = std::async(std::launch::async, [&] {
        return bool(Libraries::GnmDriver::AcquireSubmissionRetirement(
            gate, [&] {
                auto ticket = queue.CaptureCompletionTicket();
                (++captures == 1 ? captured_first : captured_retry).set_value();
                return ticket;
            }, [&] { return queue.IssuedSequence(); },
            [&] { finish_after_submit_done = submit_done_complete.load(); }, stop.get_token()));
    });
    captured_first.get_future().wait();
    auto boundary = gate.BeginGuardedBoundary();
    queue.EnqueueBoundary([complete = std::move(boundary.complete), &submit_done_complete] {
        // Actual Liverpool SubmitDone flushes/on-submits before completing this callback.
        submit_done_complete = true;
        complete();
    });
    boundary.issuance.unlock();
    queue.CompleteSubmission(sequence);
    // If the waiter won the race before the callback it captures a fresh idle ticket; otherwise
    // its first ticket is sufficient. Drain idle tickets without ever holding issuance.
    const auto deadline = std::chrono::steady_clock::now() + 1s;
    while (retire.wait_for(1ms) != std::future_status::ready &&
           std::chrono::steady_clock::now() < deadline) {
        queue.DrainReadyBoundaries();
    }
    const bool completed = retire.wait_for(0ms) == std::future_status::ready;
    stop.request_stop();
    EXPECT_TRUE(completed);
    EXPECT_TRUE(retire.get());
    EXPECT_TRUE(finish_after_submit_done.load());
}

struct RetirementSnapshot {
    bool requires_retirement;
    MemoryRangeIdentity::Snapshot identity;
};

TEST(OptimisticMemoryRetirement, AllocatorIsReleasedBeforeWaitingForIssuedGuestWork) {
    std::mutex allocator;
    std::unique_lock lock{allocator};
    bool locked_during_wait{};
    auto admission = AcquireOptimisticMemoryRetirement(
        lock, [] { return RetirementSnapshot{true, {}}; },
        [&](bool retire) {
            locked_during_wait = lock.owns_lock();
            EXPECT_TRUE(retire);
            return std::optional<int>{1};
        }, [](const auto&) { return true; });
    EXPECT_FALSE(locked_during_wait);
    EXPECT_TRUE(lock.owns_lock());
    EXPECT_TRUE(admission);
}

TEST(OptimisticMemoryRetirement, ReplacedBackingRequiresFreshCaptureBeforeMutation) {
    std::mutex allocator;
    std::unique_lock lock{allocator};
    MemoryRangeIdentity identity;
    ASSERT_TRUE(identity.Stamp(0, 16));
    unsigned acquisitions{};
    auto admission = AcquireOptimisticMemoryRetirement(
        lock, [&] { return RetirementSnapshot{true, *identity.Capture(0, 16)}; },
        [&](bool) {
            if (++acquisitions == 1) {
                EXPECT_TRUE(identity.Stamp(0, 16)); // Same metadata, different backing owner.
            }
            return std::optional<int>{1};
        }, [&](const auto& old) { return identity.Matches(old.identity); });
    EXPECT_TRUE(admission);
    EXPECT_EQ(acquisitions, 2u);
}

TEST(OptimisticMemoryRetirement, UnrelatedFreshAllocationDoesNotForceGlobalRetry) {
    std::mutex allocator;
    std::unique_lock lock{allocator};
    MemoryRangeIdentity identity;
    ASSERT_TRUE(identity.Stamp(0, 16));
    unsigned acquisitions{};
    auto admission = AcquireOptimisticMemoryRetirement(
        lock, [&] { return RetirementSnapshot{true, *identity.Capture(0, 16)}; },
        [&](bool) {
            ++acquisitions;
            EXPECT_TRUE(identity.Stamp(32, 16));
            return std::optional<int>{1};
        }, [&](const auto& old) { return identity.Matches(old.identity); });
    EXPECT_TRUE(admission);
    EXPECT_EQ(acquisitions, 1u);
}

TEST(OptimisticMemoryRetirement, CancellationDoesNotReclassifyOrMutate) {
    std::mutex allocator;
    std::unique_lock lock{allocator};
    unsigned captures{};
    unsigned validations{};
    auto admission = AcquireOptimisticMemoryRetirement(
        lock, [&] { ++captures; return RetirementSnapshot{true, {}}; },
        [](bool) { return std::optional<int>{}; },
        [&](const auto&) { ++validations; return false; });
    EXPECT_FALSE(admission);
    EXPECT_EQ(captures, 1u);
    EXPECT_EQ(validations, 0u);
}

TEST(OptimisticMemoryRetirement, FreshMappingNeverDropsAllocatorOrWaitsOnSubmissions) {
    std::mutex allocator;
    std::unique_lock lock{allocator};
    auto admission = AcquireOptimisticMemoryRetirement(
        lock, [] { return RetirementSnapshot{false, {}}; },
        [&](bool retire) {
            EXPECT_FALSE(retire);
            EXPECT_TRUE(lock.owns_lock());
            return std::optional<int>{1};
        }, [](const auto&) { ADD_FAILURE() << "Fresh lease cannot need revalidation"; return false; });
    EXPECT_TRUE(admission);
}

struct ReleaseArea { std::uint64_t size; bool free; };

TEST(PhysicalReleasePlan, UncheckedFreeHoleDoesNotConsumeLaterAllocationOutsideRequest) {
    const std::map<std::uint64_t, ReleaseArea> areas{{0, {4, false}}, {4, {4, true}},
                                                  {8, {4, false}}};
    const auto plan = PlanPhysicalMemoryRelease(areas, 0, 8, false,
                                               [](const auto& area) { return area.free; });
    ASSERT_TRUE(plan);
    EXPECT_EQ(*plan, (std::vector<MemoryReleaseSpan>{{0, 4}}));
}

TEST(PhysicalReleasePlan, AliasStartingInsideRequestedPhysicalRangeClipsAtRequestedEnd) {
    EXPECT_EQ(IntersectMemoryReleaseRanges(0, 8, 4, 8),
              (std::optional<MemoryReleaseSpan>{{4, 4}}));
}

TEST(PhysicalReleasePlan, CheckedHoleRejectsWithoutPublishingPrefix) {
    const std::map<std::uint64_t, ReleaseArea> areas{{0, {4, false}}, {4, {4, true}}};
    EXPECT_FALSE(PlanPhysicalMemoryRelease(areas, 0, 8, true,
                                          [](const auto& area) { return area.free; }));
}

TEST(PhysicalReleasePlan, TouchingDisjointAndOverflowRangesHaveNoRelease) {
    EXPECT_FALSE(IntersectMemoryReleaseRanges(4, 0, 0, 16));
    EXPECT_FALSE(IntersectMemoryReleaseRanges(0, 4, 4, 4));
    EXPECT_FALSE(IntersectMemoryReleaseRanges(0, 4, 8, 4));
    EXPECT_FALSE(IntersectMemoryReleaseRanges(std::numeric_limits<std::uint64_t>::max() - 3,
                                             8, 0, 16));
}

TEST(PhysicalReleasePlan, CheckedReleaseRequiresWholeValidPhysicalRange) {
    const std::map<std::uint64_t, ReleaseArea> areas{{4, {4, false}}, {12, {4, false}}};
    EXPECT_FALSE(PlanPhysicalMemoryRelease(areas, 0, 8, true,
                                          [](const auto& area) { return area.free; }));
    EXPECT_FALSE(PlanPhysicalMemoryRelease(areas, 4, 12, true,
                                          [](const auto& area) { return area.free; }));
    const auto partial = PlanPhysicalMemoryRelease(areas, 6, 8, false,
                                                   [](const auto& area) { return area.free; });
    ASSERT_TRUE(partial);
    EXPECT_EQ(*partial, (std::vector<MemoryReleaseSpan>{{6, 2}, {12, 2}}));
}

TEST(PhysicalReleasePlan, CheckedContiguousFragmentsAreClippedWithoutExtraMutation) {
    const std::map<std::uint64_t, ReleaseArea> areas{{0, {4, false}}, {4, {8, false}}};
    const auto plan = PlanPhysicalMemoryRelease(areas, 2, 6, true,
                                               [](const auto& area) { return area.free; });
    ASSERT_TRUE(plan);
    EXPECT_EQ(*plan, (std::vector<MemoryReleaseSpan>{{2, 2}, {4, 4}}));
}

TEST(OptimisticMemoryRetirement, FreshAllocationAndAscUnblockOldWorkBeforeBackingRetirement) {
    std::mutex allocator;
    MemoryRangeIdentity identity;
    ASSERT_TRUE(identity.Stamp(0, 16));
    Libraries::GnmDriver::SubmissionGate gate;
    AmdGpu::SubmissionCompletionQueue queue;
    const auto old_work = queue.IssueSubmission();
    std::promise<void> first_ticket;
    std::atomic<unsigned> captures{};
    std::atomic<unsigned> finishes{};
    std::atomic<bool> old_backing_live{true};
    MemoryRetirementProvider provider{[&](std::stop_token stop)
                                          -> std::optional<MemoryRetirementProvider::Token> {
        auto admission = Libraries::GnmDriver::AcquireSubmissionRetirement(
            gate, [&] {
                auto ticket = queue.CaptureCompletionTicket();
                if (++captures == 1) first_ticket.set_value();
                return ticket;
            }, [&] { return queue.IssuedSequence(); }, [&] { ++finishes; }, stop);
        if (!admission) return std::nullopt;
        return MemoryRetirementProvider::Token{[guard = std::move(*admission)] {}};
    }};
    auto retire = std::async(std::launch::async, [&] {
        std::unique_lock lock{allocator};
        auto admission = AcquireOptimisticMemoryRetirement(
            lock, [&] { return RetirementSnapshot{true, *identity.Capture(0, 16)}; },
            [&](bool required) { return required ? provider.Retire() : provider.Lease(); },
            [&](const auto& old) { return identity.Matches(old.identity); });
        if (admission) old_backing_live = false;
        return bool(admission);
    });
    const auto started = first_ticket.get_future().wait_for(1s);
    EXPECT_EQ(started, std::future_status::ready);
    std::unique_lock fresh_allocation{allocator, std::try_to_lock};
    const bool progressed = fresh_allocation.owns_lock();
    EXPECT_TRUE(progressed);
    if (progressed) {
        auto lease = provider.Lease();
        EXPECT_TRUE(lease);
        EXPECT_TRUE(identity.Stamp(32, 16));
        fresh_allocation.unlock();
        auto issuance = gate.Enter();
        const auto asc = queue.IssueSubmission();
        issuance.unlock();
        // The new ASC produces the value needed by an older WAIT_REG_MEM. The queue preserves
        // contiguous CP retirement even though this newer compute task finishes first.
        queue.CompleteSubmission(asc);
        EXPECT_TRUE(old_backing_live.load());
        EXPECT_EQ(finishes.load(), 0u);
        queue.CompleteSubmission(old_work);
    }
    const auto deadline = std::chrono::steady_clock::now() + 1s;
    while (progressed && retire.wait_for(1ms) != std::future_status::ready &&
           std::chrono::steady_clock::now() < deadline) {
        queue.DrainReadyBoundaries();
    }
    const bool completed = retire.wait_for(0ms) == std::future_status::ready;
    provider.CloseAndWait(); // Cleanup also cancels a deliberately broken acquisition fixture.
    EXPECT_TRUE(completed);
    EXPECT_TRUE(retire.get());
    EXPECT_FALSE(old_backing_live.load());
    EXPECT_EQ(finishes.load(), 1u);
}

} // namespace
} // namespace Core
