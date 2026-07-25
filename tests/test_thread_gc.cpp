// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <thread>

#include <gtest/gtest.h>

#include "core/libraries/kernel/threads/thread_gc.h"

namespace Libraries::Kernel {
namespace {

TEST(ThreadGcAccounting, ReachesAndLeavesCollectionThreshold) {
    ThreadGcAccounting accounting;

    for (std::size_t index = 0; index < 4; ++index) {
        accounting.MarkQueued();
    }
    EXPECT_FALSE(accounting.NeedsCollection(5));

    accounting.MarkQueued();
    EXPECT_TRUE(accounting.NeedsCollection(5));

    accounting.MarkCollected();
    EXPECT_FALSE(accounting.NeedsCollection(5));
}

TEST(ThreadGcAccounting, CountsConcurrentQueueAndCollectionUpdates) {
    constexpr std::size_t WorkerCount = 4;
    constexpr std::size_t UpdatesPerWorker = 1000;
    ThreadGcAccounting accounting;
    std::array<std::jthread, WorkerCount> workers;

    for (auto& worker : workers) {
        worker = std::jthread([&] {
            for (std::size_t index = 0; index < UpdatesPerWorker; ++index) {
                accounting.MarkQueued();
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    EXPECT_EQ(accounting.PendingCount(), WorkerCount * UpdatesPerWorker);

    for (auto& worker : workers) {
        worker = std::jthread([&] {
            for (std::size_t index = 0; index < UpdatesPerWorker; ++index) {
                accounting.MarkCollected();
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    EXPECT_EQ(accounting.PendingCount(), 0);
}

} // namespace
} // namespace Libraries::Kernel
