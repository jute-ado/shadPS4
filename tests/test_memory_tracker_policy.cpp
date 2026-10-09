// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/buffer_cache/memory_tracker_policy.h"

TEST(MemoryTrackerPolicy, CpuWritePreservesGpuOwnedBytesWhenOptionalReadbacksAreDisabled) {
    EXPECT_TRUE(VideoCore::ShouldFlushGpuModifiedPageOnCpuWrite(true));
}

TEST(MemoryTrackerPolicy, CpuWriteSkipsFlushWhenPageHasNoGpuOwnedBytes) {
    EXPECT_FALSE(VideoCore::ShouldFlushGpuModifiedPageOnCpuWrite(false));
}
