// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/kernel/threads/thread_reuse.h"

namespace Libraries::Kernel {
namespace {

struct TrackedResource {
    explicit TrackedResource(int* destructions_) : destructions{destructions_} {}

    ~TrackedResource() {
        ++*destructions;
    }

    int* destructions;
};

struct ReusableThreadStorage {
    explicit ReusableThreadStorage(int* thread_destructions_)
        : thread_destructions{thread_destructions_} {}

    ~ReusableThreadStorage() {
        ++*thread_destructions;
    }

    TrackedResource* sleepqueue{};
    int* thread_destructions;
};

TEST(ThreadReuse, ReleasesDetachedSleepQueueBeforeCachingThreadStorage) {
    int resource_destructions = 0;
    int thread_destructions = 0;
    alignas(ReusableThreadStorage) std::byte storage[sizeof(ReusableThreadStorage)];
    auto* thread = std::construct_at(reinterpret_cast<ReusableThreadStorage*>(storage),
                                     &thread_destructions);
    thread->sleepqueue = new TrackedResource{&resource_destructions};

    DestroyThreadForReuse(thread);

    EXPECT_EQ(resource_destructions, 1);
    EXPECT_EQ(thread_destructions, 1);
}

} // namespace
} // namespace Libraries::Kernel
