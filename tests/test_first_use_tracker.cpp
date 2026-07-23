// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "common/first_use_tracker.h"

TEST(FirstUseTracker, ReturnsTrueOnlyForTheFirstUseOfAValue) {
    Common::FirstUseTracker<4> tracker;

    EXPECT_TRUE(tracker.IsFirstUse(2));
    EXPECT_FALSE(tracker.IsFirstUse(2));
    EXPECT_FALSE(tracker.IsFirstUse(2));
}

TEST(FirstUseTracker, TracksValuesIndependently) {
    Common::FirstUseTracker<4> tracker;

    EXPECT_TRUE(tracker.IsFirstUse(0));
    EXPECT_TRUE(tracker.IsFirstUse(3));
    EXPECT_FALSE(tracker.IsFirstUse(0));
    EXPECT_FALSE(tracker.IsFirstUse(3));
}

TEST(FirstUseTracker, RejectsValuesOutsideTheTrackedRange) {
    Common::FirstUseTracker<4> tracker;

    EXPECT_FALSE(tracker.IsFirstUse(4));
    EXPECT_FALSE(tracker.IsFirstUse(100));
}

TEST(FirstUseTracker, AllowsOnlyOneConcurrentFirstUse) {
    Common::FirstUseTracker<4> tracker;
    std::atomic_int first_uses = 0;
    std::vector<std::jthread> threads;
    threads.reserve(16);

    for (int i = 0; i < 16; ++i) {
        threads.emplace_back([&] { first_uses += tracker.IsFirstUse(1); });
    }
    threads.clear();

    EXPECT_EQ(first_uses, 1);
}
