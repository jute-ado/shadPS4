// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/pending_operation_queue.h"

namespace {

struct TestOperation {
    int value{};
    int ready_tick{};
};

using Queue = Vulkan::PendingOperationQueue<TestOperation>;

TEST(PendingOperationQueue, StopsAtFirstOperationThatIsNotReady) {
    Queue queue;
    queue.Push({.value = 1, .ready_tick = 1});
    queue.Push({.value = 2, .ready_tick = 3});
    queue.Push({.value = 3, .ready_tick = 2});

    std::vector<int> observed;
    const auto drained = queue.DrainReady(
        [](const TestOperation& op) { return op.ready_tick <= 2; },
        [&](TestOperation&& op) { observed.push_back(op.value); });

    EXPECT_EQ(drained, 1u);
    EXPECT_EQ(observed, std::vector<int>({1}));
    EXPECT_EQ(queue.Size(), 2u);
}

TEST(PendingOperationQueue, CallbackCanEnqueueWithoutDeadlockOrReordering) {
    Queue queue;
    queue.Push({.value = 1, .ready_tick = 0});
    queue.Push({.value = 2, .ready_tick = 0});

    std::vector<int> observed;
    const auto drained = queue.DrainReady(
        [](const TestOperation&) { return true; }, [&](TestOperation&& op) {
            observed.push_back(op.value);
            if (op.value == 1) {
                queue.Push({.value = 3, .ready_tick = 0});
            }
        });

    EXPECT_EQ(drained, 3u);
    EXPECT_EQ(observed, std::vector<int>({1, 2, 3}));
    EXPECT_TRUE(queue.Empty());
}

TEST(PendingOperationQueue, ConcurrentProducersDoNotLoseOperations) {
    Queue queue;
    constexpr int ProducerCount = 4;
    constexpr int OperationsPerProducer = 500;
    std::vector<std::jthread> producers;
    for (int producer = 0; producer < ProducerCount; ++producer) {
        producers.emplace_back([&, producer] {
            for (int operation = 0; operation < OperationsPerProducer; ++operation) {
                queue.Push({.value = producer * OperationsPerProducer + operation,
                            .ready_tick = 0});
            }
        });
    }
    producers.clear();

    std::atomic<int> observed{};
    const auto drained = queue.DrainReady(
        [](const TestOperation&) { return true; },
        [&](TestOperation&&) { observed.fetch_add(1, std::memory_order_relaxed); });

    EXPECT_EQ(drained, ProducerCount * OperationsPerProducer);
    EXPECT_EQ(observed.load(std::memory_order_relaxed), ProducerCount * OperationsPerProducer);
    EXPECT_TRUE(queue.Empty());
}

TEST(PendingOperationQueue, SchedulerUsesTheOrdinaryQueueForBothEnqueueAndDrain) {
    std::ifstream header{U3_SCHEDULER_HEADER_PATH};
    std::ifstream source{U3_SCHEDULER_SOURCE_PATH};
    ASSERT_TRUE(header);
    ASSERT_TRUE(source);
    const std::string header_text{std::istreambuf_iterator<char>{header}, {}};
    const std::string source_text{std::istreambuf_iterator<char>{source}, {}};

    EXPECT_NE(header_text.find("PendingOperationQueue<PendingOp> pending_ops"), std::string::npos);
    EXPECT_NE(header_text.find("pending_ops.Push("), std::string::npos);
    EXPECT_NE(source_text.find("pending_ops.DrainReady("), std::string::npos);
    EXPECT_EQ(source_text.find("PopPendingOperations() {\n    std::unique_lock lk(priority_pending_ops_mutex)"),
              std::string::npos);
}

} // namespace
