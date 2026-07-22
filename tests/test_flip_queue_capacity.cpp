// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/videoout/flip_queue_capacity.h"

namespace Libraries::VideoOut {
namespace {

TEST(FlipQueueCapacity, AcceptsDisplayBufferFlipBelowCapacity) {
    EXPECT_TRUE(CanQueueFlip(MaxPendingFlips - 1, 0));
}

TEST(FlipQueueCapacity, RejectsDisplayBufferFlipAtCapacity) {
    EXPECT_FALSE(CanQueueFlip(MaxPendingFlips, 0));
}

TEST(FlipQueueCapacity, RejectsDisplayBufferFlipAboveCapacity) {
    EXPECT_FALSE(CanQueueFlip(MaxPendingFlips + 1, 0));
}

TEST(FlipQueueCapacity, BlankFlipBypassesDisplayBufferCapacity) {
    EXPECT_TRUE(CanQueueFlip(MaxPendingFlips, -1));
    EXPECT_TRUE(CanQueueFlip(MaxPendingFlips + 1, -1));
}

} // namespace
} // namespace Libraries::VideoOut
