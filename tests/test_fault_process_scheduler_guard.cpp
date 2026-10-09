// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// This model covers the earlier ProcessFaultBuffer completion path separately
// from DrainReadyFaults.  Waiting for a previous readback slot can wrap the
// stream, which submits the command buffer and may otherwise drain unrelated
// owner callbacks before that slot is collected.

#include <cassert>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace {

class SchedulerModel {
public:
    using Callback = std::function<void()>;

    class ScopedPendingOperationDrainSuppression {
    public:
        explicit ScopedPendingOperationDrainSuppression(SchedulerModel& scheduler_)
            : scheduler{&scheduler_} {
            ++scheduler->suppression_depth;
        }

        ScopedPendingOperationDrainSuppression(const ScopedPendingOperationDrainSuppression&) =
            delete;
        ScopedPendingOperationDrainSuppression& operator=(
            const ScopedPendingOperationDrainSuppression&) = delete;

        ~ScopedPendingOperationDrainSuppression() {
            assert(scheduler->suppression_depth > 0U);
            --scheduler->suppression_depth;
        }

    private:
        SchedulerModel* scheduler;
    };

    void DeferOperation(Callback callback) {
        pending.emplace_back(std::move(callback));
    }

    void Wait(std::uint64_t /*wait_tick*/) {
        events.emplace_back("wait");
        Flush();
    }

    void CollectCompletedFaults() {
        events.emplace_back("collect");
    }

    void Flush() {
        events.emplace_back("flush");
        PopPendingOperations();
    }

    void PopPendingOperations() {
        events.emplace_back("pop");
        if (suppression_depth != 0) {
            return;
        }
        while (!pending.empty()) {
            auto callback = std::move(pending.front());
            pending.pop_front();
            callback();
        }
    }

    [[nodiscard]] const std::vector<std::string>& Events() const {
        return events;
    }

private:
    std::deque<Callback> pending;
    std::vector<std::string> events;
    unsigned suppression_depth{};
};

struct ProcessObservation {
    bool retirement_ran_before_collection{};
    std::vector<std::string> events;
};

ProcessObservation RunProcessFaultBufferPath(SchedulerModel& scheduler,
                                             bool suppress_pending_operation_drain) {
    bool collection_active = true;
    bool retirement_ran_before_collection = false;
    scheduler.DeferOperation([&] { retirement_ran_before_collection |= collection_active; });

    if (suppress_pending_operation_drain) {
        SchedulerModel::ScopedPendingOperationDrainSuppression guard{scheduler};
        scheduler.Wait(1);
        scheduler.CollectCompletedFaults();
    } else {
        scheduler.Wait(1);
        scheduler.CollectCompletedFaults();
    }

    collection_active = false;
    scheduler.PopPendingOperations();
    return {retirement_ran_before_collection, scheduler.Events()};
}

enum class CallbackPhase {
    BeforeCollection,
    DuringResidency,
    AfterResidency,
};

struct OwnerBoundaryObservation {
    CallbackPhase callback_phase{};
    std::vector<std::string> events;
};

OwnerBoundaryObservation RunOwnerBoundaryPath(SchedulerModel& scheduler) {
    CallbackPhase current_phase = CallbackPhase::BeforeCollection;
    CallbackPhase callback_phase = CallbackPhase::BeforeCollection;
    scheduler.DeferOperation([&] {
        // The callback must not run while either fault-manager mutation window is active.
        callback_phase = current_phase;
    });

    {
        SchedulerModel::ScopedPendingOperationDrainSuppression process_guard{scheduler};
        scheduler.Wait(1);
        scheduler.CollectCompletedFaults();
    }

    // ProcessFaultBuffer has published the next fault-buffer command.  DrainReadyFaults then
    // has its own suppression scope while residency may wait/flush again.
    current_phase = CallbackPhase::DuringResidency;
    {
        SchedulerModel::ScopedPendingOperationDrainSuppression residency_guard{scheduler};
        scheduler.Wait(2);
    }

    // This is the proposed safe point: the owner boundary is outside both fault-manager scopes,
    // so pending callbacks are released before the following garbage-collection work.
    current_phase = CallbackPhase::AfterResidency;
    scheduler.PopPendingOperations();
    return {callback_phase, scheduler.Events()};
}

TEST(FaultProcessSchedulerGuardModel, UnscopedWaitRunsRetirementBeforeReadbackCollection) {
    SchedulerModel scheduler;
    const auto observation = RunProcessFaultBufferPath(scheduler, false);

    // RED observation: ProcessFaultBuffer's slot wait can submit and drain
    // arbitrary callbacks before CollectCompletedFaults has copied the slot.
    EXPECT_TRUE(observation.retirement_ran_before_collection);
}

TEST(FaultProcessSchedulerGuardModel, ScopedWaitDefersRetirementUntilAfterReadbackCollection) {
    SchedulerModel scheduler;
    const auto observation = RunProcessFaultBufferPath(scheduler, true);

    EXPECT_FALSE(observation.retirement_ran_before_collection);
    ASSERT_EQ(observation.events,
              (std::vector<std::string>{"wait", "flush", "pop", "collect", "pop"}));
}

TEST(FaultProcessSchedulerGuardModel, NestedScopesRemainSuppressedAcrossCollection) {
    SchedulerModel scheduler;
    bool callback_ran{};
    scheduler.DeferOperation([&] { callback_ran = true; });

    {
        SchedulerModel::ScopedPendingOperationDrainSuppression outer{scheduler};
        {
            SchedulerModel::ScopedPendingOperationDrainSuppression inner{scheduler};
            scheduler.Wait(1);
            scheduler.CollectCompletedFaults();
            EXPECT_FALSE(callback_ran);
        }
        scheduler.PopPendingOperations();
        EXPECT_FALSE(callback_ran);
    }

    scheduler.PopPendingOperations();
    EXPECT_TRUE(callback_ran);
}

TEST(FaultProcessSchedulerGuardModel,
     OwnerBoundaryDrainRunsOnlyAfterBothFaultMutationWindowsClose) {
    SchedulerModel scheduler;
    const auto observation = RunOwnerBoundaryPath(scheduler);

    EXPECT_EQ(observation.callback_phase, CallbackPhase::AfterResidency);
    ASSERT_EQ(observation.events, (std::vector<std::string>{"wait", "flush", "pop", "collect",
                                                            "wait", "flush", "pop", "pop"}));
}

#if defined(SHADPS4_ENABLE_FAULT_RESIDENCY_GUARD_WIRING) &&                                        \
    defined(SHADPS4_FAULT_RESIDENCY_SOURCE_PATH)

std::string ReadSource(const char* path) {
    std::ifstream input{path};
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

TEST(FaultProcessSchedulerGuardWiring,
     ProcessFaultBufferSuppressesDrainAcrossWaitAndReadbackCollection) {
    const auto source = ReadSource(SHADPS4_FAULT_RESIDENCY_SOURCE_PATH);
    const auto function = source.find("void FaultManager::ProcessFaultBuffer()");
    ASSERT_NE(function, std::string::npos);
    const auto wait = source.find("scheduler.Wait(wait_tick)", function);
    ASSERT_NE(wait, std::string::npos);
    const auto collect = source.find("CollectCompletedFaults(current_area)", wait);
    ASSERT_NE(collect, std::string::npos);
    const auto function_end = source.find("\n}\n\n", function);
    ASSERT_NE(function_end, std::string::npos);
    const auto guard = source.find("ScopedPendingOperationDrainSuppression", function);
    ASSERT_NE(guard, std::string::npos);
    EXPECT_LT(guard, function_end);
    EXPECT_LT(guard, wait);
    EXPECT_LT(wait, collect);
}

#endif

} // namespace
