// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/buffer_cache/buffer_barrier_policy.h"
#include "video_core/buffer_cache/buffer_write_range_tracker.h"
#include "video_core/renderer_vulkan/buffer_barrier_merge.h"

using VideoCore::NeedsBufferBarrier;
using VideoCore::BufferShaderAccess;
using VideoCore::GetBufferShaderAccess;
using VideoCore::NeedsGpuModifiedReadBarrier;
using VideoCore::ShouldSnapshotGpuModifiedComputeControl;
using VideoCore::BufferWriteRangeTracker;

TEST(BufferBarrierPolicy, UnrelatedReadDoesNotConsumePendingWrite) {
    BufferWriteRangeTracker writes;
    writes.Add(100, 48);

    EXPECT_FALSE(writes.Intersects(0, 64));
    writes.Subtract(0, 64);
    EXPECT_TRUE(writes.Intersects(100, 48));
}

TEST(BufferBarrierPolicy, ReadConsumesOnlyItsOverlappingWriteRange) {
    BufferWriteRangeTracker writes;
    writes.Add(100, 48);

    writes.Subtract(120, 10);
    EXPECT_TRUE(writes.Intersects(100, 20));
    EXPECT_FALSE(writes.Intersects(120, 10));
    EXPECT_TRUE(writes.Intersects(130, 18));
}

TEST(BufferBarrierPolicy, AdjacentWritesAreMerged) {
    BufferWriteRangeTracker writes;
    writes.Add(100, 20);
    writes.Add(120, 28);

    EXPECT_TRUE(writes.Intersects(100, 48));
    writes.Subtract(100, 48);
    EXPECT_FALSE(writes.Intersects(100, 48));
}

TEST(BufferBarrierPolicy, AccessTransitionsPreserveUnrelatedWriterHistory) {
    BufferWriteRangeTracker writes;

    EXPECT_FALSE(writes.Transition(true, 100, 48));
    EXPECT_FALSE(writes.Transition(false, 0, 64));
    EXPECT_TRUE(writes.Transition(false, 120, 10));
    EXPECT_FALSE(writes.Transition(false, 120, 10));
    EXPECT_TRUE(writes.Intersects(100, 20));
    EXPECT_TRUE(writes.Intersects(130, 18));
}

TEST(BufferBarrierPolicy, WritableDescriptorRetainsShaderReadAccess) {
    EXPECT_EQ(GetBufferShaderAccess(false), BufferShaderAccess::ReadOnly);
    EXPECT_EQ(GetBufferShaderAccess(true), BufferShaderAccess::ReadWrite);
}

TEST(BufferBarrierPolicy, RepeatedReadOnlyAccessNeedsNoBarrier) {
    EXPECT_FALSE(NeedsBufferBarrier(true, true, false));
}

TEST(BufferBarrierPolicy, RepeatedWriteAccessNeedsBarrier) {
    EXPECT_TRUE(NeedsBufferBarrier(true, true, true));
}

TEST(BufferBarrierPolicy, ReadWriteAccessNeedsBarrierWhenRepeated) {
    EXPECT_TRUE(NeedsBufferBarrier(true, true, true));
}

TEST(BufferBarrierPolicy, StateTransitionNeedsBarrier) {
    EXPECT_TRUE(NeedsBufferBarrier(false, true, true));
    EXPECT_TRUE(NeedsBufferBarrier(true, false, false));
}

TEST(BufferBarrierPolicy, GpuModifiedReadNeedsFallbackWithoutWriteDependency) {
    EXPECT_TRUE(NeedsGpuModifiedReadBarrier(true, false, false));
    EXPECT_FALSE(NeedsGpuModifiedReadBarrier(false, false, false));
    EXPECT_FALSE(NeedsGpuModifiedReadBarrier(true, true, false));
    EXPECT_FALSE(NeedsGpuModifiedReadBarrier(true, false, true));
}

TEST(BufferBarrierPolicy, SmallGpuModifiedComputeControlUsesGpuSnapshot) {
    EXPECT_TRUE(ShouldSnapshotGpuModifiedComputeControl(true, true, false, false, 48));
    EXPECT_TRUE(ShouldSnapshotGpuModifiedComputeControl(true, true, false, false, 352));
    EXPECT_TRUE(ShouldSnapshotGpuModifiedComputeControl(true, true, false, false, 512));
    EXPECT_FALSE(ShouldSnapshotGpuModifiedComputeControl(true, true, false, false, 513));
    EXPECT_FALSE(ShouldSnapshotGpuModifiedComputeControl(false, true, false, false, 48));
    EXPECT_FALSE(ShouldSnapshotGpuModifiedComputeControl(true, false, false, false, 48));
    EXPECT_FALSE(ShouldSnapshotGpuModifiedComputeControl(true, true, true, false, 48));
    EXPECT_FALSE(ShouldSnapshotGpuModifiedComputeControl(true, true, false, true, 48));
}

TEST(BufferBarrierPolicy, AliasedIndirectAndShaderBarriersMergeAllDependencies) {
    std::vector<vk::BufferMemoryBarrier2> barriers{
        vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eDrawIndirect,
            .dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead,
            .offset = 64,
            .size = 4096,
        },
    };
    const vk::BufferMemoryBarrier2 shader_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eVertexShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .offset = 64,
        .size = 4096,
    };

    EXPECT_TRUE(Vulkan::TryMergeBufferBarrier(barriers, shader_barrier));
    ASSERT_EQ(barriers.size(), 1U);
    EXPECT_EQ(barriers.front().dstStageMask,
              vk::PipelineStageFlagBits2::eDrawIndirect |
                  vk::PipelineStageFlagBits2::eVertexShader);
    EXPECT_EQ(barriers.front().dstAccessMask,
              vk::AccessFlagBits2::eIndirectCommandRead |
                  vk::AccessFlagBits2::eShaderRead);
}

TEST(BufferBarrierPolicy, DifferentRangesRemainIndependent) {
    std::vector<vk::BufferMemoryBarrier2> barriers{
        vk::BufferMemoryBarrier2{.offset = 0, .size = 64},
    };
    const vk::BufferMemoryBarrier2 other{.offset = 64, .size = 64};

    EXPECT_FALSE(Vulkan::TryMergeBufferBarrier(barriers, other));
}
