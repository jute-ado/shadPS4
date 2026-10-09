// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/pending_operation_queue.h"

TEST(PendingOperationQueue, DrainsReadyOperationsInSubmissionOrder) {
    VideoCore::PendingOperationQueue<std::function<void()>> queue;
    std::vector<int> calls;
    queue.Emplace([&] { calls.push_back(1); }, 3);
    queue.Emplace([&] { calls.push_back(2); }, 5);
    queue.Emplace([&] { calls.push_back(3); }, 8);

    queue.PopReady([](u64 tick) { return tick <= 5; });

    EXPECT_EQ(calls, (std::vector<int>{1, 2}));
    EXPECT_EQ(queue.Size(), 1u);
}

TEST(PendingOperationQueue, CallbackMayDeferAnotherReadyOperation) {
    VideoCore::PendingOperationQueue<std::function<void()>> queue;
    std::vector<int> calls;
    queue.Emplace(
        [&] {
            calls.push_back(1);
            queue.Emplace([&] { calls.push_back(2); }, 7);
        },
        7);

    queue.PopReady([](u64) { return true; });

    EXPECT_EQ(calls, (std::vector<int>{1, 2}));
    EXPECT_EQ(queue.Size(), 0u);
}

TEST(PendingOperationQueue, ConcurrentDeferralsAreNotLost) {
    VideoCore::PendingOperationQueue<std::function<void()>> queue;
    constexpr int ThreadCount = 4;
    constexpr int OperationsPerThread = 128;
    std::atomic<int> calls{};
    std::vector<std::jthread> threads;
    for (int thread = 0; thread < ThreadCount; ++thread) {
        threads.emplace_back([&] {
            for (int operation = 0; operation < OperationsPerThread; ++operation) {
                queue.Emplace([&] { ++calls; }, 1);
            }
        });
    }
    threads.clear();

    EXPECT_EQ(queue.Size(), ThreadCount * OperationsPerThread);
    queue.PopReady([](u64) { return true; });
    EXPECT_EQ(calls, ThreadCount * OperationsPerThread);
    EXPECT_EQ(queue.Size(), 0u);
}
