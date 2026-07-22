// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/pm4_memory_wait.h"

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
