// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/readonly_stream_snapshot.h"

namespace {

enum class WatchOwner {
    None,
    Persistent,
    SnapshotTransaction,
};

struct MutatingGuestRange {
    void Reserve() {
        events.push_back("reserve");
    }

    template <typename Snapshot>
    void Transaction(Snapshot&& snapshot) {
        const bool original_dirty = dirty;
        const bool original_writeable = writeable;
        const WatchOwner original_watch_owner = watch_owner;
        VideoCore::WithReadonlyStreamPageTransaction(
            [&] {
                events.push_back("lock");
                range_mutex.lock();
                snapshot_locked.store(true, std::memory_order_release);
                dirty = false;
                writeable = false;
                watch_owner = WatchOwner::SnapshotTransaction;
                events.push_back("protect");
            },
            std::forward<Snapshot>(snapshot),
            [&] {
                dirty = original_dirty;
                writeable = original_writeable;
                watch_owner = original_watch_owner;
                events.push_back("restore");
                snapshot_locked.store(false, std::memory_order_release);
                range_mutex.unlock();
                events.push_back("unlock");
            });
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

struct CrossManagerGuestRange {
    static constexpr size_t PageSize = 16;
    static constexpr size_t ManagerSize = 4 * PageSize;
    static constexpr size_t RangeBegin = ManagerSize - PageSize - 1;
    static constexpr size_t RangeSize = 2 * PageSize + 2;

    struct Page {
        bool dirty{true};
        bool writeable{true};
        WatchOwner owner{WatchOwner::None};
    };

    struct Manager {
        std::array<Page, 4> pages;
        bool locked{};
    };

    CrossManagerGuestRange() {
        managers[0].pages[3] = {
            .dirty = false, .writeable = false, .owner = WatchOwner::Persistent};
        managers[1].pages[0] = {
            .dirty = false, .writeable = false, .owner = WatchOwner::Persistent};
    }

    template <typename Func>
    void ForEachTouchedPage(Func&& func) {
        const size_t range_end = RangeBegin + RangeSize;
        for (size_t address = (RangeBegin / PageSize) * PageSize; address < range_end;
             address += PageSize) {
            const size_t manager = address / ManagerSize;
            const size_t page = (address % ManagerSize) / PageSize;
            std::invoke(func, manager, page, managers[manager].pages[page]);
        }
    }

    void Reserve() {
        events.push_back("reserve");
    }

    void Acquire() {
        managers[0].locked = true;
        events.push_back("lock_0");
        managers[1].locked = true;
        events.push_back("lock_1");
        ForEachTouchedPage([&](size_t manager, size_t page, Page& state) {
            if (!state.dirty) {
                return;
            }
            dirty_pages.emplace_back(manager, page);
            state.dirty = false;
            state.writeable = false;
            state.owner = WatchOwner::SnapshotTransaction;
        });
        events.push_back("protect");
    }

    void Restore() {
        events.push_back("restore");
        for (const auto [manager, page] : dirty_pages) {
            managers[manager].pages[page] = {
                .dirty = true, .writeable = true, .owner = WatchOwner::None};
        }
        managers[1].locked = false;
        events.push_back("unlock_1");
        managers[0].locked = false;
        events.push_back("unlock_0");
    }

    void ObserveSnapshot() {
        events.push_back("snapshot");
        snapshot_saw_all_protected = true;
        ForEachTouchedPage([&](size_t, size_t, const Page& state) {
            snapshot_saw_all_protected &= !state.dirty && !state.writeable;
        });
    }

    [[noreturn]] void ThrowDuringSnapshot() {
        events.push_back("snapshot_throw");
        throw std::runtime_error{"snapshot failed"};
    }

    void Commit() {
        ++commit_count;
        commit_saw_unlocked = !managers[0].locked && !managers[1].locked;
        events.push_back("commit");
    }

    [[nodiscard]] bool HasOriginalPageState() const {
        const Page dirty_page{.dirty = true, .writeable = true, .owner = WatchOwner::None};
        const Page clean_page{.dirty = false, .writeable = false, .owner = WatchOwner::Persistent};
        const auto same = [](const Page& lhs, const Page& rhs) {
            return lhs.dirty == rhs.dirty && lhs.writeable == rhs.writeable &&
                   lhs.owner == rhs.owner;
        };
        return same(managers[0].pages[2], dirty_page) && same(managers[0].pages[3], clean_page) &&
               same(managers[1].pages[0], clean_page) && same(managers[1].pages[1], dirty_page);
    }

    std::array<Manager, 2> managers;
    std::vector<std::pair<size_t, size_t>> dirty_pages;
    std::vector<std::string_view> events;
    bool snapshot_saw_all_protected{};
    bool commit_saw_unlocked{};
    int commit_count{};
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
        [&](auto&& snapshot) { memory.Transaction(std::forward<decltype(snapshot)>(snapshot)); },
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

TEST(ReadonlyStreamSnapshot, RestoresDirtyAndCleanPagesAcrossUnalignedManagerBoundary) {
    CrossManagerGuestRange memory;

    VideoCore::StageReadonlyStreamSnapshot(
        [&] {
            memory.Reserve();
            return std::span<int>{};
        },
        [&](auto&& snapshot) {
            VideoCore::WithReadonlyStreamPageTransaction([&] { memory.Acquire(); },
                                                         std::forward<decltype(snapshot)>(snapshot),
                                                         [&] { memory.Restore(); });
        },
        [&](std::span<int>) { memory.ObserveSnapshot(); }, [&] { memory.Commit(); });

    EXPECT_TRUE(memory.snapshot_saw_all_protected);
    EXPECT_TRUE(memory.HasOriginalPageState());
    EXPECT_TRUE(memory.commit_saw_unlocked);
    EXPECT_EQ(memory.commit_count, 1);
    EXPECT_EQ(memory.events,
              (std::vector<std::string_view>{"reserve", "lock_0", "lock_1", "protect", "snapshot",
                                             "restore", "unlock_1", "unlock_0", "commit"}));
}

TEST(ReadonlyStreamSnapshot, RestoresAndUnlocksBeforePropagatingSnapshotException) {
    CrossManagerGuestRange memory;

    EXPECT_THROW(VideoCore::StageReadonlyStreamSnapshot(
                     [&] {
                         memory.Reserve();
                         return std::span<int>{};
                     },
                     [&](auto&& snapshot) {
                         VideoCore::WithReadonlyStreamPageTransaction(
                             [&] { memory.Acquire(); }, std::forward<decltype(snapshot)>(snapshot),
                             [&] { memory.Restore(); });
                     },
                     [&](std::span<int>) { memory.ThrowDuringSnapshot(); },
                     [&] { memory.Commit(); }),
                 std::runtime_error);

    EXPECT_TRUE(memory.HasOriginalPageState());
    EXPECT_FALSE(memory.managers[0].locked);
    EXPECT_FALSE(memory.managers[1].locked);
    EXPECT_EQ(memory.commit_count, 0);
    EXPECT_EQ(memory.events,
              (std::vector<std::string_view>{"reserve", "lock_0", "lock_1", "protect",
                                             "snapshot_throw", "restore", "unlock_1", "unlock_0"}));
}
