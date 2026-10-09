// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <condition_variable>
#include <fstream>
#include <future>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/buffer_registration_index.h"

namespace VideoCore {

struct BufferRegistrationIndexTestAccess {
    template <typename Id, typename Mutex>
    static Mutex& GetMutex(BufferRegistrationIndex<Id, Mutex>& index) {
        return index.mutex;
    }
};

namespace {
using namespace std::chrono_literals;

// Observe the real helper's lock attempts before they block. A missing lock fails immediately
// when the operation returns; no concurrent unsynchronized container mutation is needed for RED.
struct ObservedMutex {
    void lock() {
        {
            std::scoped_lock lock{notification};
            ++attempts;
        }
        changed.notify_all();
        backing.lock();
    }
    void unlock() { backing.unlock(); }
    bool try_lock() { return backing.try_lock(); }
    void Finished() {
        {
            std::scoped_lock lock{notification};
            finished = true;
        }
        changed.notify_all();
    }
    bool WaitForAttempt(unsigned expected) {
        std::unique_lock lock{notification};
        return changed.wait_for(lock, 1s, [&] { return attempts >= expected || finished; }) &&
               attempts >= expected;
    }
    unsigned AttemptCount() {
        std::scoped_lock lock{notification};
        return attempts;
    }
    std::mutex backing;
    std::mutex notification;
    std::condition_variable changed;
    unsigned attempts{};
    bool finished{};
};

template <typename Operation>
void RequiresSharedIndexLock(Operation&& operation) {
    BufferRegistrationIndex<u32, ObservedMutex> index;
    index.Add(0x1000, 0x1000, 7);
    auto& mutex = BufferRegistrationIndexTestAccess::GetMutex(index);
    std::unique_lock held{mutex};
    const auto next_attempt = mutex.AttemptCount() + 1;
    auto worker = std::async(std::launch::async, [&] {
        operation(index);
        mutex.Finished();
    });
    const bool attempted = mutex.WaitForAttempt(next_attempt);
    const bool blocked = worker.wait_for(0ms) == std::future_status::timeout;
    // Always release the fixture lock before assertions or joining, including RED/failure paths.
    held.unlock();
    worker.get();
    EXPECT_TRUE(attempted);
    EXPECT_TRUE(blocked);
}

TEST(BufferRegistrationIndex, CpuQueryUsesTheSharedIndexLock) {
    RequiresSharedIndexLock([](auto& index) { EXPECT_TRUE(index.Intersects(0x1000, 1)); });
}

TEST(BufferRegistrationIndex, OwnerInsertionUsesTheSharedIndexLock) {
    RequiresSharedIndexLock([](auto& index) { index.Add(0x3000, 0x1000, 8); });
}

TEST(BufferRegistrationIndex, OwnerRemovalUsesTheSharedIndexLock) {
    RequiresSharedIndexLock([](auto& index) { index.Subtract(0x1000, 0x1000); });
}

TEST(BufferRegistrationIndex, PreservesRangeQueriesAndPartialRemoval) {
    BufferRegistrationIndex<u32> index;
    EXPECT_FALSE(index.Intersects(0x1000, 1));
    index.Add(0x1000, 0x1000, 7);
    EXPECT_FALSE(index.Intersects(0xfff, 1));
    EXPECT_TRUE(index.Intersects(0x1000, 1));
    EXPECT_TRUE(index.Intersects(0x1fff, 1));
    EXPECT_FALSE(index.Intersects(0x2000, 1));
    index.Subtract(0x1400, 0x400);
    EXPECT_TRUE(index.Intersects(0x13ff, 1));
    EXPECT_FALSE(index.Intersects(0x1400, 0x400));
    EXPECT_TRUE(index.Intersects(0x1800, 1));
    index.Subtract(0x1000, 0x1000);
    EXPECT_FALSE(index.Intersects(0x1000, 0x1000));
}

TEST(BufferRegistrationIndex, OwnerEnumerationDoesNotHoldLockAcrossCallbacks) {
    BufferRegistrationIndex<u32, ObservedMutex> index;
    index.Add(0x1000, 0x1000, 7);
    auto& mutex = BufferRegistrationIndexTestAccess::GetMutex(index);
    std::vector<std::tuple<u64, u64, u32>> visited;
    index.OwnerForEachInRange(0x1400, 0x200, [&](u64 begin, u64 end, u32 id) {
        const bool unlocked = mutex.try_lock();
        if (unlocked) mutex.unlock();
        EXPECT_TRUE(unlocked);
        visited.emplace_back(begin, end, id);
    });
    EXPECT_EQ(visited, (std::vector<std::tuple<u64, u64, u32>>{{0x1400, 0x1600, 7}}));
}

std::string ReadSource(const char* path) {
    std::ifstream input{path};
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

TEST(BufferRegistrationIndex, ProductionUsesGuardedIndexAndOwnerOnlyEnumeration) {
    const auto header = ReadSource(BUFFER_CACHE_HEADER_PATH);
    const auto source = ReadSource(BUFFER_CACHE_SOURCE_PATH);
    EXPECT_NE(header.find("BufferRegistrationIndex<BufferId> buffer_ranges;"), std::string::npos);
    EXPECT_NE(header.find("buffer_ranges.OwnerForEachInRange("), std::string::npos);
    EXPECT_EQ(header.find("buffer_ranges.ForEachInRange("), std::string::npos);
    EXPECT_NE(source.find("return buffer_ranges.Intersects(addr, size);"), std::string::npos);
    EXPECT_NE(source.find("buffer_ranges.Add(buffer.CpuAddr(), buffer.SizeBytes(), buffer_id);"),
              std::string::npos);
    EXPECT_NE(source.find("buffer_ranges.Subtract(buffer.CpuAddr(), buffer.SizeBytes());"),
              std::string::npos);
}

} // namespace
} // namespace VideoCore
