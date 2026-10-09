// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/buffer_readback_window.h"

namespace {

constexpr VAddr BufferBase = 4_MB;
constexpr u64 BufferSize = 1_MB;
constexpr u64 WindowSize = 512_KB;

TEST(BufferReadbackWindow, WidensScatteredFaultsToStableAlignedWindows) {
    const auto first = VideoCore::PlanBufferReadbackWindow(
        BufferBase, BufferSize, BufferBase + 100_KB, 4_KB, WindowSize);
    ASSERT_TRUE(first.valid);
    EXPECT_EQ(first.download_address, BufferBase);
    EXPECT_EQ(first.download_size, WindowSize);
    EXPECT_EQ(first.request_address, BufferBase + 100_KB);
    EXPECT_EQ(first.request_size, 4_KB);

    const auto second = VideoCore::PlanBufferReadbackWindow(
        BufferBase, BufferSize, BufferBase + 700_KB, 4_KB, WindowSize);
    ASSERT_TRUE(second.valid);
    EXPECT_EQ(second.download_address, BufferBase + WindowSize);
    EXPECT_EQ(second.download_size, WindowSize);
    EXPECT_EQ(second.request_address, BufferBase + 700_KB);
    EXPECT_EQ(second.request_size, 4_KB);
}

TEST(BufferReadbackWindow, CoversLargeRequestsAndClampsToTheBackingBuffer) {
    const auto large = VideoCore::PlanBufferReadbackWindow(
        BufferBase, BufferSize, BufferBase + 128_KB, 768_KB, WindowSize);
    ASSERT_TRUE(large.valid);
    EXPECT_LE(large.download_address, large.request_address);
    EXPECT_GE(large.download_address + large.download_size,
              large.request_address + large.request_size);
    EXPECT_GE(large.download_size, large.request_size);

    const auto small_buffer = VideoCore::PlanBufferReadbackWindow(
        BufferBase, 128_KB, BufferBase + 64_KB, 4_KB, WindowSize);
    ASSERT_TRUE(small_buffer.valid);
    EXPECT_EQ(small_buffer.download_address, BufferBase);
    EXPECT_EQ(small_buffer.download_size, 128_KB);
}

TEST(BufferReadbackWindow, RejectsRangesThatCannotBelongToTheSelectedBuffer) {
    EXPECT_FALSE(VideoCore::PlanBufferReadbackWindow(
                     BufferBase, BufferSize, BufferBase - 1, 4_KB, WindowSize)
                     .valid);
    EXPECT_FALSE(VideoCore::PlanBufferReadbackWindow(
                     BufferBase, BufferSize, BufferBase + BufferSize - 1, 2, WindowSize)
                     .valid);
    EXPECT_FALSE(VideoCore::PlanBufferReadbackWindow(BufferBase, BufferSize, BufferBase, 0,
                                                     WindowSize)
                     .valid);
    EXPECT_FALSE(VideoCore::PlanBufferReadbackWindow(BufferBase, BufferSize, BufferBase, 4_KB, 0)
                     .valid);
    EXPECT_FALSE(VideoCore::PlanBufferReadbackWindow(
                     std::numeric_limits<VAddr>::max() - 1, 4, std::numeric_limits<VAddr>::max(),
                     1, WindowSize)
                     .valid);
}

} // namespace
