// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <vector>

#include <gtest/gtest.h>

#include "video_core/amdgpu/pixel_pipe_stat_control.h"

using AmdGpu::PixelPipeStatControl;

TEST(PixelPipeStatControl, DecodesCounterStrideAndSparseInstanceMask) {
    constexpr u32 counter_id = 5;
    constexpr u32 stride_encoding = 2;
    constexpr u64 instance_mask = (1ULL << 0) | (1ULL << 3) | (1ULL << 20) | (1ULL << 23);
    const u32 low = (counter_id << 3) | (stride_encoding << 9) |
                    (static_cast<u32>(instance_mask) << 11);
    const u32 high = static_cast<u32>(instance_mask >> 21);

    const auto control = PixelPipeStatControl::Decode(low, high);

    EXPECT_EQ(control.counter_id, counter_id);
    EXPECT_EQ(control.stride_bytes, 16);
    EXPECT_EQ(control.instance_enable_mask, instance_mask);
    EXPECT_EQ(control.EnabledInstanceCount(), 4);
}

TEST(PixelPipeStatControl, VisitsOnlyEnabledInstancesAtConfiguredStride) {
    constexpr u64 instance_mask = (1ULL << 1) | (1ULL << 4) | (1ULL << 7);
    const auto control = PixelPipeStatControl::Decode(
        (1U << 9) | (static_cast<u32>(instance_mask) << 11),
        static_cast<u32>(instance_mask >> 21));
    std::vector<u32> byte_offsets;

    control.ForEachEnabledInstance(8, [&](u32, u32 byte_offset) {
        byte_offsets.push_back(byte_offset);
    });

    EXPECT_EQ(byte_offsets, (std::vector<u32>{8, 32, 56}));
}
