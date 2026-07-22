// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/kernel/threads/semaphore_policy.h"

namespace Libraries::Kernel {
namespace {

TEST(KernelSemaphorePolicy, WaitCountMustBePositiveAndFitSemaphore) {
    EXPECT_FALSE(IsValidSemaphoreWaitCount(-1, 4));
    EXPECT_FALSE(IsValidSemaphoreWaitCount(0, 4));
    EXPECT_TRUE(IsValidSemaphoreWaitCount(1, 4));
    EXPECT_TRUE(IsValidSemaphoreWaitCount(4, 4));
    EXPECT_FALSE(IsValidSemaphoreWaitCount(5, 4));
}

TEST(KernelSemaphorePolicy, SignalCountMustBePositiveAndFitAvailableCapacity) {
    EXPECT_FALSE(IsValidSemaphoreSignalCount(-1, 1, 4));
    EXPECT_FALSE(IsValidSemaphoreSignalCount(0, 1, 4));
    EXPECT_TRUE(IsValidSemaphoreSignalCount(1, 1, 4));
    EXPECT_TRUE(IsValidSemaphoreSignalCount(3, 1, 4));
    EXPECT_FALSE(IsValidSemaphoreSignalCount(4, 1, 4));
}

TEST(KernelSemaphorePolicy, CancelCountCanResetOrFitSemaphore) {
    EXPECT_TRUE(IsValidSemaphoreCancelCount(-1, 4));
    EXPECT_TRUE(IsValidSemaphoreCancelCount(0, 4));
    EXPECT_TRUE(IsValidSemaphoreCancelCount(4, 4));
    EXPECT_FALSE(IsValidSemaphoreCancelCount(5, 4));
}

TEST(KernelSemaphorePolicy, RemainingTimeoutNeverWrapsPastZero) {
    EXPECT_EQ(RemainingSemaphoreTimeout(100, 40), 60u);
    EXPECT_EQ(RemainingSemaphoreTimeout(100, 100), 0u);
    EXPECT_EQ(RemainingSemaphoreTimeout(100, 101), 0u);
    EXPECT_EQ(RemainingSemaphoreTimeout(100, 1'000'000), 0u);
}

} // namespace
} // namespace Libraries::Kernel
