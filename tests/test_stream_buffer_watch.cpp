// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/stream_buffer_watch.h"

TEST(StreamBufferWatch, ReusesLastWatchWhenCommitsShareSubmissionTick) {
    std::array<VideoCore::StreamBufferWatch, 2> watches{{
        {.tick = 42, .upper_bound = 64},
        {},
    }};

    auto* watch = VideoCore::FindLastCommittedStreamBufferWatch(watches, 1, 42);

    ASSERT_EQ(watch, &watches[0]);
    watch->upper_bound = 96;
    EXPECT_EQ(watches[0].upper_bound, 96);
}

TEST(StreamBufferWatch, IgnoresRecycledNextSlotWhenTicksCoincide) {
    std::array<VideoCore::StreamBufferWatch, 2> watches{{
        {.tick = 42, .upper_bound = 64},
        {.tick = 42, .upper_bound = 16},
    }};

    auto* watch = VideoCore::FindLastCommittedStreamBufferWatch(watches, 1, 42);

    ASSERT_EQ(watch, &watches[0]);
    watch->upper_bound = 96;
    EXPECT_EQ(watches[0].upper_bound, 96);
    EXPECT_EQ(watches[1].upper_bound, 16);
}
