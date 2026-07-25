// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <thread>
#include <unordered_set>

#include <gtest/gtest.h>

#include "core/libraries/kernel/threads/thread_cache.h"

namespace Libraries::Kernel {
namespace {

TEST(ThreadAllocationPolicy, RejectsTheConfiguredLimitAndBeyond) {
    EXPECT_TRUE(CanAllocateThread(0, 100));
    EXPECT_TRUE(CanAllocateThread(99, 100));
    EXPECT_FALSE(CanAllocateThread(100, 100));
    EXPECT_FALSE(CanAllocateThread(101, 100));
}

TEST(ThreadCache, IsBoundedAndLastInFirstOut) {
    BoundedThreadCache<int, 2> cache;
    int first = 1;
    int second = 2;
    int rejected = 3;

    EXPECT_TRUE(cache.TryPush(&first));
    EXPECT_TRUE(cache.TryPush(&second));
    EXPECT_FALSE(cache.TryPush(&rejected));
    EXPECT_EQ(cache.TryPop(), &second);
    EXPECT_EQ(cache.TryPop(), &first);
    EXPECT_EQ(cache.TryPop(), nullptr);
}

TEST(ThreadCache, SerializesConcurrentAccess) {
    constexpr std::size_t EntryCount = 128;
    BoundedThreadCache<int, EntryCount> cache;
    std::array<int, EntryCount> entries{};
    std::array<std::jthread, 4> producers;

    for (std::size_t producer = 0; producer < producers.size(); ++producer) {
        producers[producer] = std::jthread([&, producer] {
            const std::size_t begin = producer * EntryCount / producers.size();
            const std::size_t end = (producer + 1) * EntryCount / producers.size();
            for (std::size_t index = begin; index < end; ++index) {
                EXPECT_TRUE(cache.TryPush(&entries[index]));
            }
        });
    }
    for (auto& producer : producers) {
        producer.join();
    }

    std::unordered_set<int*> popped;
    while (int* entry = cache.TryPop()) {
        popped.insert(entry);
    }
    EXPECT_EQ(popped.size(), EntryCount);
}

} // namespace
} // namespace Libraries::Kernel
