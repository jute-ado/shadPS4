// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/buffer_cache/stream_buffer_retirement.h"

TEST(StreamBufferRetirement, PreservesUnreachedTailAcrossConsecutiveWraps) {
    // Generation A ended at the ring boundary with two retirement watches. Generation B reached
    // only the first watch before wrapping, so A's tail must retire before its ledger is reused.
    EXPECT_EQ(VideoCore::PendingStreamBufferWatchesBeforeRotation(2, 1), 1);
}

TEST(StreamBufferRetirement, FirstOrFullyRetiredGenerationCanRotateImmediately) {
    EXPECT_EQ(VideoCore::PendingStreamBufferWatchesBeforeRotation(std::nullopt, 0), 0);
    EXPECT_EQ(VideoCore::PendingStreamBufferWatchesBeforeRotation(2, 2), 0);
}
