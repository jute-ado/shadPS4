// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_compute_dispatch_chunking.h"

using Vulkan::ComputeDispatchChunk3D;
using Vulkan::ComputeDispatchChunk3DAt;
using Vulkan::ComputeDispatchChunk3DCount;
using Vulkan::ComputeDispatchChunkShape;
using Vulkan::WatchdogSafeComputeChunkShape;

TEST(ComputeDispatchChunking, LeavesOrdinaryShadersAsOneDispatch) {
    const auto shape = WatchdogSafeComputeChunkShape(false, 20, 12, 12);
    EXPECT_EQ(shape, (ComputeDispatchChunkShape{20, 12, 12}));
    EXPECT_EQ(ComputeDispatchChunk3DCount(20, 12, 12, shape), 1U);
}

TEST(ComputeDispatchChunking, BoundsDynamicMaskLoopsByTotalWorkgroupBudget) {
    constexpr u32 budget = 128;
    const auto shape = WatchdogSafeComputeChunkShape(true, 20, 12, 12, budget);
    EXPECT_EQ(shape, (ComputeDispatchChunkShape{16, 8, 1}));
    EXPECT_LE(shape.x * shape.y * shape.z, budget);
    EXPECT_EQ(ComputeDispatchChunk3DCount(20, 12, 12, shape), 48U);

    EXPECT_EQ(ComputeDispatchChunk3DAt(20, 12, 12, shape, 0),
              (ComputeDispatchChunk3D{0, 0, 0, 16, 8, 1}));
    EXPECT_EQ(ComputeDispatchChunk3DAt(20, 12, 12, shape, 47),
              (ComputeDispatchChunk3D{16, 8, 11, 4, 4, 1}));
}

TEST(ComputeDispatchChunking, UsesTheFullBudgetForOneDimensionalDispatches) {
    constexpr u32 budget = 128;
    const auto shape = WatchdogSafeComputeChunkShape(true, 16384, 1, 1, budget);

    EXPECT_EQ(shape, (ComputeDispatchChunkShape{128, 1, 1}));
    EXPECT_EQ(ComputeDispatchChunk3DCount(16384, 1, 1, shape), 128U);
}

TEST(ComputeDispatchChunking, KeepsProductionChunksBelowTheValidatedWatchdogBudget) {
    const auto volume = WatchdogSafeComputeChunkShape(true, 20, 12, 64);
    EXPECT_EQ(volume, (ComputeDispatchChunkShape{20, 12, 4}));
    EXPECT_EQ(ComputeDispatchChunk3DCount(20, 12, 64, volume), 16U);

    const auto surface = WatchdogSafeComputeChunkShape(true, 120, 68, 1);
    EXPECT_EQ(surface, (ComputeDispatchChunkShape{64, 16, 1}));
    EXPECT_EQ(ComputeDispatchChunk3DCount(120, 68, 1, surface), 10U);

    const auto row = WatchdogSafeComputeChunkShape(true, 16384, 1, 1);
    EXPECT_EQ(row, (ComputeDispatchChunkShape{1024, 1, 1}));
    EXPECT_EQ(ComputeDispatchChunk3DCount(16384, 1, 1, row), 16U);
}

TEST(ComputeDispatchChunking, PreservesEveryOriginalWorkgroupCoordinateExactlyOnce) {
    const auto shape = WatchdogSafeComputeChunkShape(true, 120, 68, 1, 128);
    ASSERT_EQ(shape, (ComputeDispatchChunkShape{16, 8, 1}));
    ASSERT_EQ(ComputeDispatchChunk3DCount(120, 68, 1, shape), 72U);

    std::vector<u8> visits(120 * 68);
    for (u32 index = 0; index < 72; ++index) {
        const auto chunk = ComputeDispatchChunk3DAt(120, 68, 1, shape, index);
        for (u32 y = chunk.base_y; y < chunk.base_y + chunk.count_y; ++y) {
            for (u32 x = chunk.base_x; x < chunk.base_x + chunk.count_x; ++x) {
                ++visits[y * 120 + x];
            }
        }
    }
    EXPECT_TRUE(std::ranges::all_of(visits, [](u8 visits_for_group) {
        return visits_for_group == 1;
    }));
}
