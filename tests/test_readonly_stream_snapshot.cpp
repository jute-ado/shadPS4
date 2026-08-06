// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <mutex>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/readonly_stream_snapshot.h"

namespace {

enum class WatchOwner {
    None,
    SnapshotTransaction,
};

struct MutatingGuestRange {
    void Reserve() {
        events.push_back("reserve");
    }

    template <typename Snapshot>
    void Transaction(Snapshot&& snapshot) {
        events.push_back("lock");
        std::unique_lock lock{range_mutex};
        snapshot_locked.store(true, std::memory_order_release);

        const bool original_dirty = dirty;
        const bool original_writeable = writeable;
        const WatchOwner original_watch_owner = watch_owner;
        dirty = false;
        writeable = false;
        watch_owner = WatchOwner::SnapshotTransaction;
        events.push_back("protect");

        std::invoke(std::forward<Snapshot>(snapshot));

        dirty = original_dirty;
        writeable = original_writeable;
        watch_owner = original_watch_owner;
        events.push_back("restore");
        snapshot_locked.store(false, std::memory_order_release);
        lock.unlock();
        events.push_back("unlock");
    }

    void CopySnapshot(std::span<int> destination) {
        events.push_back("snapshot_begin");
        std::ranges::copy(source.begin(), source.begin() + 2, destination.begin());
        halfway.store(true, std::memory_order_release);
        while (!writer_attempted.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        if (!snapshot_locked.load(std::memory_order_acquire)) {
            while (!writer_completed.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        }
        std::ranges::copy(source.begin() + 2, source.end(), destination.begin() + 2);
        events.push_back("snapshot_end");
    }

    void MutateFromGuestThread() {
        while (!halfway.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        writer_attempted.store(true, std::memory_order_release);
        std::scoped_lock lock{range_mutex};
        source.fill(2);
        writer_completed.store(true, std::memory_order_release);
    }

    void Commit() {
        events.push_back("commit");
    }

    std::array<int, 4> source{1, 1, 1, 1};
    std::array<int, 4> staging{};
    std::mutex range_mutex;
    std::atomic<bool> halfway{};
    std::atomic<bool> writer_attempted{};
    std::atomic<bool> writer_completed{};
    std::atomic<bool> snapshot_locked{};
    bool dirty{true};
    bool writeable{true};
    WatchOwner watch_owner{WatchOwner::None};
    std::vector<std::string_view> events;
};

} // namespace

TEST(ReadonlyStreamSnapshot, OwnsACoherentSnapshotWhileGuestMutationIsAttempted) {
    MutatingGuestRange memory;
    std::jthread writer{[&] { memory.MutateFromGuestThread(); }};

    VideoCore::StageReadonlyStreamSnapshot(
        [&] {
            memory.Reserve();
            return std::span<int>{memory.staging};
        },
        [&](auto&& snapshot) {
            memory.Transaction(std::forward<decltype(snapshot)>(snapshot));
        },
        [&](std::span<int> destination) { memory.CopySnapshot(destination); },
        [&] { memory.Commit(); });
    writer.join();

    EXPECT_EQ(memory.staging, (std::array<int, 4>{1, 1, 1, 1}));
    EXPECT_EQ(memory.source, (std::array<int, 4>{2, 2, 2, 2}));
    EXPECT_TRUE(memory.writer_attempted.load());
    EXPECT_TRUE(memory.writer_completed.load());
    EXPECT_TRUE(memory.dirty);
    EXPECT_TRUE(memory.writeable);
    EXPECT_EQ(memory.watch_owner, WatchOwner::None);
    EXPECT_EQ(memory.events,
              (std::vector<std::string_view>{"reserve", "lock", "protect", "snapshot_begin",
                                             "snapshot_end", "restore", "unlock", "commit"}));
}
