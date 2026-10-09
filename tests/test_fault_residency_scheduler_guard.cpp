// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// This is a small model for the fault-residency ordering hazard.  The
// unscoped-path test characterizes the failure (Wait -> Flush -> Pop), while
// the guarded-path tests specify the production behavior. The dedicated target
// enables the source-wiring checks so the scopes cannot silently be removed.

#include <cassert>
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

    // This mirrors the current production path: Wait flushes the current
    // command buffer, and SubmitExecution then drains owner callbacks.
    void WaitForStreamWrap() {
        events.emplace_back("wait");
        Flush();
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

struct ResidencyObservation {
    bool retirement_ran_during_residency{};
    std::vector<std::string> events;
};

ResidencyObservation RunResidencyThatWrapsStream(SchedulerModel& scheduler,
                                                 bool suppress_pending_operation_drain) {
    bool residency_active = true;
    bool retirement_ran_during_residency = false;
    scheduler.DeferOperation([&] { retirement_ran_during_residency |= residency_active; });

    if (suppress_pending_operation_drain) {
        SchedulerModel::ScopedPendingOperationDrainSuppression guard{scheduler};
        scheduler.WaitForStreamWrap();
    } else {
        scheduler.WaitForStreamWrap();
    }

    residency_active = false;
    // The enclosing guest boundary is responsible for the eventual callback
    // drain after residency and command publication have completed.
    scheduler.PopPendingOperations();
    return {retirement_ran_during_residency, scheduler.Events()};
}

TEST(FaultResidencySchedulerGuardModel, UnscopedIndirectWaitRunsRetirementTooEarly) {
    SchedulerModel scheduler;
    const auto observation = RunResidencyThatWrapsStream(scheduler, false);

    // This is the RED observation: without a scope, Scheduler::Wait can flush
    // and pop arbitrary callbacks while MakeCompletedFaultsResident is active.
    EXPECT_TRUE(observation.retirement_ran_during_residency);
}

TEST(FaultResidencySchedulerGuardModel, NestableSuppressionDefersRetirementUntilTheOuterBoundary) {
    SchedulerModel scheduler;
    const auto observation = RunResidencyThatWrapsStream(scheduler, true);

    EXPECT_FALSE(observation.retirement_ran_during_residency);
    ASSERT_EQ(observation.events, (std::vector<std::string>{"wait", "flush", "pop", "pop"}));
}

TEST(FaultResidencySchedulerGuardModel, NestedScopesRemainSuppressedUntilTheLastScopeExits) {
    SchedulerModel scheduler;
    bool callback_ran{};
    scheduler.DeferOperation([&] { callback_ran = true; });

    {
        SchedulerModel::ScopedPendingOperationDrainSuppression outer{scheduler};
        {
            SchedulerModel::ScopedPendingOperationDrainSuppression inner{scheduler};
            scheduler.WaitForStreamWrap();
            EXPECT_FALSE(callback_ran);
        }
        scheduler.PopPendingOperations();
        EXPECT_FALSE(callback_ran);
    }

    scheduler.PopPendingOperations();
    EXPECT_TRUE(callback_ran);
}

#if defined(SHADPS4_ENABLE_FAULT_RESIDENCY_GUARD_WIRING) &&                                        \
    defined(SHADPS4_FAULT_RESIDENCY_SOURCE_PATH) && defined(SHADPS4_SCHEDULER_SOURCE_PATH)

std::string ReadSource(const char* path) {
    std::ifstream input{path};
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

TEST(FaultResidencySchedulerGuardWiring, FaultDrainHoldsSuppressionAcrossResidencyConsumer) {
    const auto source = ReadSource(SHADPS4_FAULT_RESIDENCY_SOURCE_PATH);
    const auto function = source.find("void FaultManager::DrainReadyFaults()");
    ASSERT_NE(function, std::string::npos);
    const auto guard = source.find("ScopedPendingOperationDrainSuppression", function);
    const auto drain = source.find("completed_faults.Drain()", function);
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(drain, std::string::npos);
    EXPECT_LT(guard, drain);
}

TEST(FaultResidencySchedulerGuardWiring, SchedulerTrailingDrainsHonorSuppressionScope) {
    const auto source = ReadSource(SHADPS4_SCHEDULER_SOURCE_PATH);
    ASSERT_FALSE(source.empty());
    constexpr std::string_view function_name = "void Scheduler::PopPendingOperations()";
    constexpr std::string_view suppressed = "IsPendingOperationDrainSuppressed()";
    const auto function = source.find(function_name);
    ASSERT_NE(function, std::string::npos);
    const auto guard = source.find(suppressed, function);
    ASSERT_NE(guard, std::string::npos);
    const auto return_statement = source.find("return;", guard);
    ASSERT_NE(return_statement, std::string::npos);
    EXPECT_LT(guard, return_statement);
}

#endif

} // namespace
