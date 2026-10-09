// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/cache_storage.h"

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <thread>

namespace {

using namespace std::chrono_literals;

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("shadps4-cache-storage-" + std::to_string(timestamp) + "-" +
                std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
        std::filesystem::create_directories(path);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    std::filesystem::path path;
};

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream input{path, std::ios::binary};
    return std::string{std::istreambuf_iterator<char>{input}, {}};
}

TEST(CacheStorage, CloseDrainsAcceptedWritesBeforeReopen) {
    TemporaryDirectory directory;
    Storage::Detail::AsyncWriteQueue queue;

    std::promise<void> first_write_started;
    std::promise<void> release_first_write;
    const auto release = release_first_write.get_future().share();

    queue.Open();
    ASSERT_TRUE(queue.Submit([&] {
        first_write_started.set_value();
        release.wait();
    }));
    first_write_started.get_future().wait();

    const std::array<std::pair<std::string, std::string>, 3> expected{{
        {"late.meta", "metadata"},
        {"late.spv", "shader-binary"},
        {"late.key", "pipeline-key"},
    }};
    for (const auto& [name, contents] : expected) {
        ASSERT_TRUE(queue.Submit([path = directory.path / name, contents] {
            std::ofstream output{path, std::ios::binary};
            output << contents;
        }));
    }

    auto close = std::async(std::launch::async, [&] { queue.Close(); });
    EXPECT_EQ(close.wait_for(20ms), std::future_status::timeout);
    release_first_write.set_value();
    ASSERT_EQ(close.wait_for(5s), std::future_status::ready);
    close.get();

    // Check synchronously before reopening. Reopen must not be able to mask an
    // early Close return by letting the old worker drain these writes later.
    for (const auto& [name, contents] : expected) {
        EXPECT_EQ(ReadFile(directory.path / name), contents);
    }

    // Reopening the queue models the next emulator run. Every write accepted by
    // the prior run must already be visible before the new run starts reading.
    queue.Open();
    std::array<std::string, expected.size()> actual{};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        ASSERT_TRUE(queue.Submit(
            [&, index] { actual[index] = ReadFile(directory.path / expected[index].first); }));
    }
    queue.Flush();
    queue.Close();

    for (std::size_t index = 0; index < expected.size(); ++index) {
        EXPECT_EQ(actual[index], expected[index].second);
    }
}

TEST(CacheStorage, SubmitRemainsNonBlockingWhileWorkerIsBusy) {
    Storage::Detail::AsyncWriteQueue queue;
    std::promise<void> first_write_started;
    std::promise<void> release_first_write;
    const auto release = release_first_write.get_future().share();
    std::atomic_bool late_write_ran{};

    queue.Open();
    ASSERT_TRUE(queue.Submit([&] {
        first_write_started.set_value();
        release.wait();
    }));
    first_write_started.get_future().wait();

    ASSERT_TRUE(queue.Submit([&] { late_write_ran.store(true, std::memory_order_release); }));
    EXPECT_FALSE(late_write_ran.load(std::memory_order_acquire));

    release_first_write.set_value();
    queue.Close();
    EXPECT_TRUE(late_write_ran.load(std::memory_order_acquire));
    EXPECT_FALSE(queue.Submit([] {}));
}

TEST(CacheStorage, FlushWaitsForAcceptedWritesWithoutClosingQueue) {
    Storage::Detail::AsyncWriteQueue queue;
    std::promise<void> first_write_started;
    std::promise<void> release_first_write;
    const auto release = release_first_write.get_future().share();
    std::atomic_bool late_write_ran{};

    queue.Open();
    ASSERT_TRUE(queue.Submit([&] {
        first_write_started.set_value();
        release.wait();
    }));
    first_write_started.get_future().wait();
    ASSERT_TRUE(queue.Submit([&] { late_write_ran.store(true, std::memory_order_release); }));

    auto flush = std::async(std::launch::async, [&] { queue.Flush(); });
    EXPECT_EQ(flush.wait_for(20ms), std::future_status::timeout);
    release_first_write.set_value();
    ASSERT_EQ(flush.wait_for(5s), std::future_status::ready);
    flush.get();

    EXPECT_TRUE(late_write_ran.load(std::memory_order_acquire));
    EXPECT_TRUE(queue.Submit([] {}));
    queue.Close();
}

TEST(CacheStorage, ConcurrentCloseWaiterCompletesAcrossReopen) {
    Storage::Detail::AsyncWriteQueue queue;
    std::promise<void> write_started;
    std::promise<void> release_write;
    const auto release = release_write.get_future().share();

    queue.Open();
    ASSERT_TRUE(queue.Submit([&] {
        write_started.set_value();
        release.wait();
    }));
    write_started.get_future().wait();

    auto first_close = std::async(std::launch::async, [&] { queue.Close(); });
    while (queue.Submit([] {})) {
        std::this_thread::yield();
    }

    std::promise<void> second_close_started;
    auto second_close = std::async(std::launch::async, [&] {
        second_close_started.set_value();
        queue.Close();
    });
    second_close_started.get_future().wait();
    EXPECT_EQ(second_close.wait_for(20ms), std::future_status::timeout);

    auto reopen = std::async(std::launch::async, [&] { queue.Open(); });
    release_write.set_value();

    ASSERT_EQ(first_close.wait_for(5s), std::future_status::ready);
    ASSERT_EQ(reopen.wait_for(5s), std::future_status::ready);
    first_close.get();
    reopen.get();

    const auto second_close_status = second_close.wait_for(1s);
    if (second_close_status != std::future_status::ready) {
        // Unblock a regressed waiter before its std::future is destroyed, so a
        // failure reports normally instead of hanging the entire test binary.
        queue.Close();
    }
    EXPECT_EQ(second_close_status, std::future_status::ready);
    ASSERT_EQ(second_close.wait_for(5s), std::future_status::ready);
    second_close.get();
    queue.Close();
}

TEST(CacheStorage, DatabaseFinalizesArchiveOnlyAfterQueueClose) {
    std::ifstream input{SHADPS4_CACHE_STORAGE_SOURCE_PATH, std::ios::binary};
    const std::string source{std::istreambuf_iterator<char>{input}, {}};

    const auto close = source.find("io_queue.Close();");
    const auto finalize = source.find("mz_zip_writer_finalize_archive(&zip_ar);");
    ASSERT_NE(close, std::string::npos);
    ASSERT_NE(finalize, std::string::npos);
    EXPECT_LT(close, finalize);
}

} // namespace
