// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/fence_write_progress_tracker.h"
#include "video_core/amdgpu/pm4_memory_wait.h"
#include "video_core/amdgpu/wait_yield_tracker.h"

TEST(Pm4MemoryWait, PublishesGpuWritesBeforeRecheckingAnUnsatisfiedWait) {
    unsigned label = 0;
    unsigned publications = 0;

    const bool satisfied = AmdGpu::PollGpuMemoryWait([&] { return label == 1; },
                                                     [&] {
                                                         ++publications;
                                                         label = 1;
                                                     });

    EXPECT_TRUE(satisfied);
    EXPECT_EQ(publications, 1u);
}

TEST(Pm4MemoryWait, SkipsPublicationWhenTheWaitIsAlreadySatisfied) {
    unsigned label = 1;
    unsigned publications = 0;

    const bool satisfied =
        AmdGpu::PollGpuMemoryWait([&] { return label == 1; }, [&] { ++publications; });

    EXPECT_TRUE(satisfied);
    EXPECT_EQ(publications, 0u);
}

TEST(Pm4MemoryWait, ReportsAnUnsatisfiedWaitAfterPublishingCurrentGpuWrites) {
    unsigned publications = 0;

    const bool satisfied = AmdGpu::PollGpuMemoryWait([] { return false; }, [&] { ++publications; });

    EXPECT_FALSE(satisfied);
    EXPECT_EQ(publications, 1u);
}

TEST(Pm4MemoryWait, ReportsAWaitAtIncreasingYieldThresholds) {
    AmdGpu::WaitYieldTracker tracker{3};

    EXPECT_FALSE(tracker.ObserveYield());
    EXPECT_FALSE(tracker.ObserveYield());
    EXPECT_TRUE(tracker.ObserveYield());
    EXPECT_FALSE(tracker.ObserveYield());
    EXPECT_FALSE(tracker.ObserveYield());
    EXPECT_TRUE(tracker.ObserveYield());
    for (int yield = 0; yield < 5; ++yield) {
        EXPECT_FALSE(tracker.ObserveYield());
    }
    EXPECT_TRUE(tracker.ObserveYield());
    EXPECT_EQ(tracker.YieldCount(), 12u);
}

TEST(Pm4MemoryWait, CorrelatesPendingFenceWritesByAddress) {
    AmdGpu::FenceWriteProgressTracker tracker;

    tracker.Schedule(0x1234);
    tracker.Schedule(0x1234);
    tracker.Start(0x1234);
    tracker.Complete(0x1234);

    const auto progress = tracker.Snapshot(0x1234);
    EXPECT_EQ(progress.scheduled, 2u);
    EXPECT_EQ(progress.started, 1u);
    EXPECT_EQ(progress.completed, 1u);
    EXPECT_EQ(tracker.Snapshot(0x5678).scheduled, 0u);
}
