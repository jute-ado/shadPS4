// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <latch>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/eop_completion.h"
#include "video_core/amdgpu/eop_flip_tracker.h"
#include "video_core/renderer_vulkan/gpu_completion_submission.h"

namespace {

struct TestEopPacket {
    u32 data;

    void SignalFence(auto&& write_memory, auto&& signal_interrupt) const {
        write_memory(data);
        signal_interrupt();
    }
};

TEST(EventWriteEop, InterruptFlipWaitsForFollowingEopPublication) {
    std::vector<std::string> operations;
    AmdGpu::EopFlipTracker tracker;

    ASSERT_TRUE(tracker.QueueFlip(AmdGpu::DecodeFlipEopPosition(0x33),
                                  [&] { operations.emplace_back("flip"); }));
    EXPECT_TRUE(operations.empty());

    auto complete_eop = tracker.BeginEop();
    AmdGpu::PublishEop(
        TestEopPacket{.data = 1}, [&](u32) { operations.emplace_back("fence"); },
        [&] { operations.emplace_back("interrupt"); },
        [complete_eop = std::move(complete_eop)]() mutable { complete_eop(); });

    EXPECT_EQ(operations, (std::vector<std::string>{"fence", "interrupt", "flip"}));
}

TEST(EventWriteEop, PlainAndLabelFlipsFollowTheirPrecedingEopPublication) {
    for (const u32 nop_count : {0x34u, 0x39u}) {
        std::vector<std::string> operations;
        AmdGpu::EopFlipTracker tracker;

        auto complete_eop = tracker.BeginEop();
        AmdGpu::PublishEop(
            TestEopPacket{.data = 1}, [&](u32) { operations.emplace_back("fence"); },
            [&] { operations.emplace_back("interrupt"); },
            [complete_eop = std::move(complete_eop)]() mutable { complete_eop(); });

        ASSERT_TRUE(tracker.QueueFlip(AmdGpu::DecodeFlipEopPosition(nop_count),
                                      [&] { operations.emplace_back("flip"); }));
        EXPECT_EQ(operations, (std::vector<std::string>{"fence", "interrupt", "flip"}));
    }
}

TEST(EventWriteEop, RetainsEveryFlipAssociatedWithTheSameEop) {
    std::vector<int> flips;
    AmdGpu::EopFlipTracker tracker;

    ASSERT_TRUE(tracker.QueueFlip(AmdGpu::FlipEopPosition::Following,
                                  [&] { flips.emplace_back(1); }));
    ASSERT_TRUE(tracker.QueueFlip(AmdGpu::FlipEopPosition::Following,
                                  [&] { flips.emplace_back(2); }));
    auto complete_eop = tracker.BeginEop();

    EXPECT_TRUE(flips.empty());
    complete_eop();
    EXPECT_EQ(flips, (std::vector<int>{1, 2}));
}

TEST(EventWriteEop, PublishesFenceInterruptAndFlipOnlyAfterExactGpuCompletion) {
    std::vector<std::string> operations;
    Common::UniqueFunction<void> gpu_completion;

    AmdGpu::SubmitEopAtGpuCompletion(
        TestEopPacket{.data = 1},
        [&](Common::UniqueFunction<void>&& completion) {
            operations.emplace_back("submit");
            gpu_completion = std::move(completion);
        },
        [&](u32) { operations.emplace_back("fence"); },
        [&] { operations.emplace_back("interrupt"); },
        [&] { operations.emplace_back("flip"); });

    EXPECT_EQ(operations, (std::vector<std::string>{"submit"}));
    ASSERT_TRUE(gpu_completion);

    gpu_completion();

    EXPECT_EQ(operations,
              (std::vector<std::string>{"submit", "fence", "interrupt", "flip"}));
}

TEST(GpuCompletionSubmission, BindsCompletionToTickReturnedByExactSubmission) {
    std::mutex submission_mutex;
    Vulkan::GpuCompletionSubmission sequencer{submission_mutex};
    std::vector<std::string> operations;
    u64 deferred_tick{};

    const u64 submitted_tick = sequencer.Submit(
        [] {},
        [&] {
            operations.emplace_back("submit");
            return 41u;
        },
        [&](auto&&, u64 tick) {
            operations.emplace_back("defer");
            deferred_tick = tick;
        });

    EXPECT_EQ(submitted_tick, 41u);
    EXPECT_EQ(deferred_tick, submitted_tick);
    EXPECT_EQ(operations, (std::vector<std::string>{"submit", "defer"}));
}

TEST(GpuCompletionSubmission, SharesExactTickAssociationWithOrdinarySubmissions) {
    std::mutex submission_mutex;
    Vulkan::GpuCompletionSubmission sequencer{submission_mutex};
    std::atomic<u64> next_tick{1};
    std::atomic<u64> deferred_tick{};
    std::atomic<u64> ordinary_tick{};
    std::latch exact_submit_entered{1};
    std::latch ordinary_submit_attempted{1};
    std::latch release_exact_submit{1};

    std::jthread exact_submission{[&] {
        sequencer.Submit(
            [] {},
            [&] {
                const u64 tick = next_tick.fetch_add(1);
                exact_submit_entered.count_down();
                release_exact_submit.wait();
                return tick;
            },
            [&](auto&&, u64 tick) { deferred_tick = tick; });
    }};
    exact_submit_entered.wait();

    std::jthread ordinary_submission{[&] {
        ordinary_submit_attempted.count_down();
        std::scoped_lock lock{submission_mutex};
        ordinary_tick = next_tick.fetch_add(1);
    }};
    ordinary_submit_attempted.wait();

    EXPECT_EQ(ordinary_tick.load(), 0u);
    release_exact_submit.count_down();
    exact_submission.join();
    ordinary_submission.join();

    EXPECT_EQ(deferred_tick.load(), 1u);
    EXPECT_EQ(ordinary_tick.load(), 2u);
}

} // namespace
