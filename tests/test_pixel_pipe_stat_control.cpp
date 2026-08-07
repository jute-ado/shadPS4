// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/pixel_pipe_stat_control.h"

namespace AmdGpu {
namespace {

TEST(PixelPipeStatControl, DecodesCounterStrideAndSplitInstanceMask) {
    constexpr u32 CounterId = 37;
    constexpr u32 StrideEncoding = 1;
    constexpr u64 InstanceMask = (1ULL << 0) | (1ULL << 5) | (1ULL << 21) |
                                 (1ULL << 42);
    constexpr u32 ControlLo = (CounterId << 3) | (StrideEncoding << 9) |
                              (static_cast<u32>(InstanceMask) << 11);
    constexpr u32 ControlHi = static_cast<u32>(InstanceMask >> 21);

    constexpr auto control = DecodePixelPipeStatControl(ControlLo, ControlHi);

    EXPECT_EQ(control.counter_id, CounterId);
    EXPECT_EQ(control.stride_bytes, 8U);
    EXPECT_EQ(control.instance_mask, InstanceMask);
}

TEST(PixelPipeStatControl, PlansOnlyEnabledInstanceOffsets) {
    constexpr PixelPipeStatControl control{
        .counter_id = 0,
        .stride_bytes = 32,
        .instance_mask = (1ULL << 0) | (1ULL << 3) | (1ULL << 7),
    };

    constexpr auto layout = BuildPixelPipeStatLayout(control, 8);

    ASSERT_EQ(layout.count, 3U);
    EXPECT_EQ(layout.offsets[0], 0U);
    EXPECT_EQ(layout.offsets[1], 96U);
    EXPECT_EQ(layout.offsets[2], 224U);
}

TEST(PixelPipeStatControl, RecognizesCurrentFixedBaseLayout) {
    constexpr u32 ControlLo = (2U << 9) | (0xFFU << 11);
    constexpr auto control = DecodePixelPipeStatControl(ControlLo, 0);

    EXPECT_TRUE(IsFixedPixelPipeStatLayout(control, 8));
    EXPECT_FALSE(IsFixedPixelPipeStatLayout(control, 16));
}

} // namespace
} // namespace AmdGpu
