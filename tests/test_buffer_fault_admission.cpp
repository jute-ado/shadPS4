// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/buffer_cache/buffer_fault_admission.h"

TEST(BufferFaultAdmission, TracksCpuFaultWhileReplacementIsUnpublished) {
    u32 invalidation_count = 0;
    u32 dma_mark_count = 0;

    const bool tracked = VideoCore::ProcessTrackedBufferFault(
        [&] {
            ++invalidation_count;
            return true;
        },
        [&] { ++dma_mark_count; });

    EXPECT_TRUE(tracked);
    EXPECT_EQ(invalidation_count, 1);
    EXPECT_EQ(dma_mark_count, 1);
}

TEST(BufferFaultAdmission, LeavesUnknownUntrackedPageUnchanged) {
    u32 invalidation_count = 0;
    u32 dma_mark_count = 0;

    const bool tracked = VideoCore::ProcessTrackedBufferFault(
        [&] {
            ++invalidation_count;
            return false;
        },
        [&] { ++dma_mark_count; });

    EXPECT_FALSE(tracked);
    EXPECT_EQ(invalidation_count, 1);
    EXPECT_EQ(dma_mark_count, 0);
}
