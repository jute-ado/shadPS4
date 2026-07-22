// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/types.h"
#include "video_core/amdgpu/eop_completion.h"
#include "video_core/amdgpu/eop_flip_completion.h"
#include "video_core/amdgpu/submission_boundary.h"
#include "video_core/amdgpu/submission_boundary_queue.h"

namespace {

struct TestEopPacket {
    u32 data;

    void SignalFence(auto&& write_memory, auto&& signal_interrupt) const {
        write_memory(data);
        signal_interrupt();
    }
};

} // namespace

TEST(EventWriteEop, DefersFenceWriteAndInterruptUntilSubmittedGpuWorkCompletes) {
    u32 fence = 0;
    bool interrupt_signalled = false;
    std::vector<std::string> operations;
    std::function<void()> gpu_completion;

    AmdGpu::SubmitEop(
        TestEopPacket{.data = 0x12345678},
        [&](auto&& completion) {
            operations.emplace_back("defer");
            gpu_completion = std::forward<decltype(completion)>(completion);
        },
        [&] { operations.emplace_back("submit"); },
        [&](u32 data) {
            operations.emplace_back("fence");
            fence = data;
        },
        [&] {
            operations.emplace_back("interrupt");
            interrupt_signalled = true;
        });

    EXPECT_EQ(fence, 0u);
    EXPECT_FALSE(interrupt_signalled);
    EXPECT_EQ(operations, (std::vector<std::string>{"defer", "submit"}));

    ASSERT_TRUE(gpu_completion);
    gpu_completion();

    EXPECT_EQ(fence, 0x12345678u);
    EXPECT_TRUE(interrupt_signalled);
    EXPECT_EQ(operations, (std::vector<std::string>{"defer", "submit", "fence", "interrupt"}));
}

TEST(EventWriteEop, DefersFlipUntilThePrecedingEopCompletes) {
    bool flip_signalled = false;
    AmdGpu::EopFlipCompletion completion;

    completion.AttachFlip([&] { flip_signalled = true; });
    EXPECT_FALSE(flip_signalled);

    completion.CompleteEop();
    EXPECT_TRUE(flip_signalled);
}

TEST(EventWriteEop, SignalsFlipImmediatelyWhenThePrecedingEopAlreadyCompleted) {
    bool flip_signalled = false;
    AmdGpu::EopFlipCompletion completion;

    completion.CompleteEop();
    completion.AttachFlip([&] { flip_signalled = true; });

    EXPECT_TRUE(flip_signalled);
}

TEST(EventWriteEop, SignalsFlipAfterFenceWriteAndInterrupt) {
    std::vector<std::string> operations;
    std::function<void()> gpu_completion;
    AmdGpu::EopFlipCompletion completion;
    completion.AttachFlip([&] { operations.emplace_back("flip"); });

    AmdGpu::SubmitEop(
        TestEopPacket{.data = 1},
        [&](auto&& callback) { gpu_completion = std::forward<decltype(callback)>(callback); },
        [&] { operations.emplace_back("submit"); }, [&](u32) { operations.emplace_back("fence"); },
        [&] { operations.emplace_back("interrupt"); }, [&] { completion.CompleteEop(); });

    EXPECT_EQ(operations, (std::vector<std::string>{"submit"}));
    ASSERT_TRUE(gpu_completion);
    gpu_completion();

    EXPECT_EQ(operations,
              (std::vector<std::string>{"submit", "fence", "interrupt", "flip"}));
}

TEST(EventWriteEop, CompletesSubmissionBoundaryAfterEarlierEopSideEffects) {
    std::vector<std::string> operations;
    std::vector<std::function<void()>> gpu_completions;

    auto defer_completion = [&](auto&& completion) {
        gpu_completions.emplace_back(std::forward<decltype(completion)>(completion));
    };

    AmdGpu::SubmitEop(
        TestEopPacket{.data = 1}, defer_completion, [&] { operations.emplace_back("eop-submit"); },
        [&](u32) { operations.emplace_back("fence"); },
        [&] { operations.emplace_back("interrupt"); });
    AmdGpu::SubmitSubmissionBoundary([&] { operations.emplace_back("boundary"); }, defer_completion,
                                     [&] { operations.emplace_back("boundary-submit"); });

    EXPECT_EQ(operations, (std::vector<std::string>{"eop-submit", "boundary-submit"}));
    ASSERT_EQ(gpu_completions.size(), 2u);

    gpu_completions[0]();
    EXPECT_EQ(operations,
              (std::vector<std::string>{"eop-submit", "boundary-submit", "fence", "interrupt"}));

    gpu_completions[1]();
    EXPECT_EQ(operations, (std::vector<std::string>{"eop-submit", "boundary-submit", "fence",
                                                    "interrupt", "boundary"}));
}

TEST(SubmissionBoundaryQueue, PreservesConsecutiveCallbacksInFifoOrder) {
    AmdGpu::SubmissionBoundaryQueue queue;
    std::vector<int> order;

    queue.Push([&] { order.push_back(1); });
    queue.Push([&] { order.push_back(2); });

    EXPECT_FALSE(queue.Empty());
    auto first = queue.Pop();
    first();

    EXPECT_FALSE(queue.Empty());
    auto second = queue.Pop();
    second();

    EXPECT_TRUE(queue.Empty());
    EXPECT_EQ(order, (std::vector<int>{1, 2}));
}
