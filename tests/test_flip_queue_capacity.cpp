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

TEST(FlipQueueCapacity, ReservesEopFlipBeforeItsGpuInterrupt) {
    s32 pending_flips = 0;
    s32 eop_flips = 0;

    EXPECT_TRUE(ReserveFlip(pending_flips, eop_flips, 0, true));
    EXPECT_EQ(pending_flips, 1);
    EXPECT_EQ(eop_flips, 1);
}

TEST(FlipQueueCapacity, RejectedReservationDoesNotChangeAccounting) {
    s32 pending_flips = MaxPendingFlips;
    s32 eop_flips = 3;

    EXPECT_FALSE(ReserveFlip(pending_flips, eop_flips, 0, true));
    EXPECT_EQ(pending_flips, MaxPendingFlips);
    EXPECT_EQ(eop_flips, 3);
}

TEST(FlipQueueCapacity, CompletingReservedEopFlipReleasesBothCounters) {
    s32 pending_flips = 1;
    s32 eop_flips = 1;

    CompleteFlip(pending_flips, eop_flips, true);

    EXPECT_EQ(pending_flips, 0);
    EXPECT_EQ(eop_flips, 0);
}

TEST(FlipLabelGeneration, OlderPresentationCanReleaseBeforeNewerReservationCompletes) {
    FlipLabelGeneration generation;
    const auto first = generation.Reserve();
    generation.Complete(first);
    generation.Display(first);
    const auto newer = generation.Reserve();

    EXPECT_TRUE(generation.CanReleaseDisplayed());
    EXPECT_NE(first, newer);
}

TEST(FlipLabelGeneration, LatestPresentationCanReleaseItsReservation) {
    FlipLabelGeneration generation;
    const auto older = generation.Reserve();
    const auto latest = generation.Reserve();
    generation.Complete(latest);
    generation.Display(latest);

    EXPECT_TRUE(generation.CanReleaseDisplayed());
    EXPECT_NE(older, latest);
}

TEST(FlipLabelGeneration, OlderPresentationCannotHideDisplayedLatestReservation) {
    FlipLabelGeneration generation;
    const auto older = generation.Reserve();
    const auto latest = generation.Reserve();
    generation.Complete(latest);
    generation.Display(latest);
    generation.Display(older);

    EXPECT_TRUE(generation.CanReleaseDisplayed());
}

TEST(FlipLabelGeneration, OlderPresentationCannotReleaseNewerCompletedGeneration) {
    FlipLabelGeneration generation;
    const auto older = generation.Reserve();
    const auto latest = generation.Reserve();
    generation.Complete(latest);
    generation.Display(older);

    EXPECT_FALSE(generation.CanReleaseDisplayed());
}

} // namespace
} // namespace Libraries::VideoOut
