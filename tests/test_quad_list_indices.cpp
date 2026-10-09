// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <limits>

#include <gtest/gtest.h>

#include "video_core/quad_list_indices.h"

TEST(QuadListIndices, CountsOnlyCompleteQuads) {
    EXPECT_EQ(VideoCore::QuadListTriangleIndexCount(0), 0U);
    EXPECT_EQ(VideoCore::QuadListTriangleIndexCount(3), 0U);
    EXPECT_EQ(VideoCore::QuadListTriangleIndexCount(4), 6U);
    EXPECT_EQ(VideoCore::QuadListTriangleIndexCount(9), 12U);
    EXPECT_EQ(VideoCore::QuadListTriangleIndexCount(std::numeric_limits<u32>::max()), 0U);
}

TEST(QuadListIndices, GeneratesTriangleIndicesForNonIndexedDraw) {
    std::array<u32, 12> output{};
    VideoCore::GenerateQuadListTriangleIndices(output, 8);
    EXPECT_EQ(output, (std::array<u32, 12>{0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7}));
}

TEST(QuadListIndices, ConvertsSixteenBitGuestIndices) {
    const std::array<u16, 4> input{9, 4, 7, 2};
    std::array<u16, 6> output{};
    VideoCore::ConvertQuadListTriangleIndices(output, input);
    EXPECT_EQ(output, (std::array<u16, 6>{9, 4, 7, 9, 7, 2}));
}

TEST(QuadListIndices, ConvertsThirtyTwoBitGuestIndicesAndIgnoresIncompleteTail) {
    const std::array<u32, 6> input{100000, 8, 42, 77, 91, 92};
    std::array<u32, 6> output{};
    VideoCore::ConvertQuadListTriangleIndices(output, input);
    EXPECT_EQ(output, (std::array<u32, 6>{100000, 8, 42, 100000, 42, 77}));
}
