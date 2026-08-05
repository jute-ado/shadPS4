// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/buffer_cache/buffer_barrier.h"

TEST(BufferBarrier, ReusesAnIdenticalReadOnlyDependency) {
    EXPECT_FALSE(VideoCore::NeedsBufferBarrier(true, false, false));
}

TEST(BufferBarrier, EmitsDependencyWhenAccessOrStageChanges) {
    EXPECT_TRUE(VideoCore::NeedsBufferBarrier(false, false, false));
}

TEST(BufferBarrier, EmitsDependencyForRepeatedWrites) {
    EXPECT_TRUE(VideoCore::NeedsBufferBarrier(true, true, true));
}

TEST(BufferBarrier, EmitsDependencyWhenEitherAccessWrites) {
    EXPECT_TRUE(VideoCore::NeedsBufferBarrier(true, true, false));
    EXPECT_TRUE(VideoCore::NeedsBufferBarrier(true, false, true));
}
