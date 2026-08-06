// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <functional>
#include <queue>
#include <vector>

#include "video_core/renderer_vulkan/pending_operation_queue.h"

namespace {

struct PendingOperation {
    std::function<void()> callback;
    u64 gpu_tick{};
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

} // namespace
