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

TEST(SrtSnapshotEpochCoordinator, ValidSealReleasesEveryWatcherExactlyOnce) {
    SrtSnapshotEpochCoordinator coordinator;
    constexpr VAddr First = 0x242710000ULL;
    constexpr VAddr Second = First + 16_KB;
    u32 arm_count{};
    u32 release_count{};

    const u64 epoch = coordinator.BeginEpoch();
    ASSERT_NE(epoch, 0U);
    EXPECT_EQ(coordinator.BeginEpoch(), 0U);
    ASSERT_TRUE(coordinator.BeginPageCopy(epoch, First, 16_KB, [&](VAddr, u64) { ++arm_count; }));
    ASSERT_TRUE(coordinator.EndPageCopy(epoch, First, 16_KB, true));
    ASSERT_TRUE(coordinator.BeginPageCopy(epoch, Second, 16_KB, [&](VAddr, u64) { ++arm_count; }));
    ASSERT_TRUE(coordinator.EndPageCopy(epoch, Second, 16_KB, true));

    EXPECT_TRUE(coordinator.SealEpoch(epoch, [&](VAddr, u64) { ++release_count; }));
    EXPECT_EQ(arm_count, 2U);
    EXPECT_EQ(release_count, 2U);
    EXPECT_FALSE(coordinator.IsEpochValid(epoch));
}

TEST(SrtSnapshotEpochCoordinator,
     OverlappingInvalidationWaitsForCopyAndReleasesWatcherExactlyOnce) {
    SrtSnapshotEpochCoordinator coordinator;
    constexpr VAddr Address = 0x242710000ULL;
    std::atomic<u32> release_count{};

    const u64 epoch = coordinator.BeginEpoch();
    ASSERT_TRUE(coordinator.BeginPageCopy(epoch, Address, 16_KB, [](VAddr, u64) {}));

    std::promise<void> invalidation_started;
    std::promise<void> invalidation_finished;
    auto started = invalidation_started.get_future();
    auto finished = invalidation_finished.get_future();
    std::jthread writer{[&] {
        invalidation_started.set_value();
        coordinator.InvalidateRangeAndWait(Address + 64, 4, [&](VAddr, u64) { ++release_count; });
        invalidation_finished.set_value();
    }};

    ASSERT_EQ(started.wait_for(1s), std::future_status::ready);
    const auto deadline = std::chrono::steady_clock::now() + 1s;
    while (coordinator.IsEpochValid(epoch) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    EXPECT_FALSE(coordinator.IsEpochValid(epoch));
    EXPECT_EQ(finished.wait_for(0ms), std::future_status::timeout);

    EXPECT_FALSE(coordinator.EndPageCopy(epoch, Address, 16_KB, true));
    EXPECT_EQ(finished.wait_for(1s), std::future_status::ready);
    EXPECT_EQ(release_count, 1U);
    EXPECT_FALSE(coordinator.SealEpoch(epoch, [&](VAddr, u64) { ++release_count; }));
    EXPECT_EQ(release_count, 1U);
}

TEST(SrtSnapshotEpochCoordinator, UnrelatedInvalidationDoesNotCancelEpoch) {
    SrtSnapshotEpochCoordinator coordinator;
    constexpr VAddr Address = 0x242710000ULL;
    u32 release_count{};

    const u64 epoch = coordinator.BeginEpoch();
    ASSERT_TRUE(coordinator.BeginPageCopy(epoch, Address, 16_KB, [](VAddr, u64) {}));
    ASSERT_TRUE(coordinator.EndPageCopy(epoch, Address, 16_KB, true));
    coordinator.InvalidateRangeAndWait(Address + 32_KB, 4, [&](VAddr, u64) { ++release_count; });

    EXPECT_TRUE(coordinator.IsEpochValid(epoch));
    EXPECT_TRUE(coordinator.SealEpoch(epoch, [&](VAddr, u64) { ++release_count; }));
    EXPECT_EQ(release_count, 1U);
}

} // namespace
} // namespace VideoCore
