// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include "video_core/srt_snapshot_epoch_coordinator.h"

namespace VideoCore {
namespace {

using namespace std::chrono_literals;

TEST(SrtSnapshotEpochCoordinator,
     PreUnmapCancellationWaitsForCopyAndReleasesTheWatcherExactlyOnce) {
    SrtSnapshotEpochCoordinator coordinator;
    constexpr VAddr Address = 0x242710000ULL;
    constexpr u64 Size = 16_KB;
    std::atomic<u32> arm_count{};
    std::atomic<u32> release_count{};

    const u64 epoch = coordinator.BeginEpoch();
    ASSERT_NE(epoch, 0U);
    ASSERT_TRUE(coordinator.BeginPageCopy(epoch, Address, Size, [&](VAddr address, u64 size) {
        EXPECT_EQ(address, Address);
        EXPECT_EQ(size, Size);
        ++arm_count;
    }));
    EXPECT_EQ(arm_count, 1U);

    std::promise<void> cancellation_started;
    std::promise<void> cancellation_finished;
    auto started = cancellation_started.get_future();
    auto finished = cancellation_finished.get_future();
    std::jthread unmap_thread{[&] {
        cancellation_started.set_value();
        coordinator.InvalidateRangeAndWait(Address, Size, [&](VAddr address, u64 size) {
            EXPECT_EQ(address, Address);
            EXPECT_EQ(size, Size);
            ++release_count;
        });
        cancellation_finished.set_value();
    }};

    EXPECT_EQ(started.wait_for(1s), std::future_status::ready);
    const auto invalidation_deadline = std::chrono::steady_clock::now() + 1s;
    while (coordinator.IsEpochValid(epoch) &&
           std::chrono::steady_clock::now() < invalidation_deadline) {
        std::this_thread::yield();
    }
    EXPECT_FALSE(coordinator.IsEpochValid(epoch));
    EXPECT_EQ(finished.wait_for(0ms), std::future_status::timeout);
    EXPECT_EQ(release_count, 0U);

    EXPECT_FALSE(coordinator.EndPageCopy(epoch, Address, Size, true));
    EXPECT_EQ(finished.wait_for(1s), std::future_status::ready);
    EXPECT_EQ(release_count, 1U);

    EXPECT_FALSE(coordinator.SealEpoch(epoch, [&](VAddr, u64) { ++release_count; }));
    EXPECT_EQ(release_count, 1U);
}

} // namespace
} // namespace VideoCore
