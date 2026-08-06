// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <queue>
#include <vector>

#include "video_core/renderer_vulkan/pending_operation_queue.h"

namespace {

struct PendingOperation {
    std::function<void()> callback;
    std::uint64_t gpu_tick{};
};

TEST(PendingOperationQueue, ExtractsReadyCallbacksBeforeAllowingReentrantEnqueue) {
    std::queue<PendingOperation> pending;
    std::vector<int> calls;
    pending.push({
        .callback = [&] {
            calls.push_back(1);
            pending.push({.callback = [&] { calls.push_back(2); }, .gpu_tick = 7});
        },
        .gpu_tick = 7,
    });

    auto ready = Vulkan::TakeReadyPendingOperations(
        pending, [](const PendingOperation& operation) { return operation.gpu_tick <= 7; });

    ASSERT_TRUE(pending.empty());
    ASSERT_EQ(ready.size(), 1);
    ready.front().callback();
    EXPECT_EQ(calls, std::vector<int>{1});
    ASSERT_EQ(pending.size(), 1);

    ready = Vulkan::TakeReadyPendingOperations(
        pending, [](const PendingOperation& operation) { return operation.gpu_tick <= 7; });
    ASSERT_EQ(ready.size(), 1);
    ready.front().callback();
    EXPECT_EQ(calls, (std::vector<int>{1, 2}));
    EXPECT_TRUE(pending.empty());
}

TEST(PendingOperationQueue, LeavesTheFirstIncompleteOperationQueued) {
    std::queue<PendingOperation> pending;
    pending.push({.gpu_tick = 5});
    pending.push({.gpu_tick = 9});

    const auto ready = Vulkan::TakeReadyPendingOperations(
        pending, [](const PendingOperation& operation) { return operation.gpu_tick <= 5; });

    ASSERT_EQ(ready.size(), 1);
    ASSERT_EQ(pending.size(), 1);
    EXPECT_EQ(pending.front().gpu_tick, 9);
}

TEST(PendingOperationQueue, ReentrantDrainPreservesOlderCallbackOrder) {
    std::queue<PendingOperation> pending;
    std::vector<int> calls;
    bool is_draining = false;
    std::function<void()> drain;
    drain = [&] {
        Vulkan::DrainReadyPendingOperations(
            pending, is_draining,
            [](const PendingOperation& operation) { return operation.gpu_tick <= 7; },
            [](PendingOperation& operation) { operation.callback(); });
    };
    pending.push({
        .callback = [&] {
            calls.push_back(1);
            pending.push({.callback = [&] { calls.push_back(3); }, .gpu_tick = 7});
            drain();
        },
        .gpu_tick = 7,
    });
    pending.push({.callback = [&] { calls.push_back(2); }, .gpu_tick = 7});

    drain();

    EXPECT_EQ(calls, (std::vector<int>{1, 2, 3}));
    EXPECT_FALSE(is_draining);
    EXPECT_TRUE(pending.empty());
}

} // namespace
