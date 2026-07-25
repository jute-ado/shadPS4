// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/buffer_barrier_policy.h"

TEST(BufferBarrierPolicy, ReadOnlyBindingRequiresShaderReadAccess) {
    EXPECT_EQ(Vulkan::ShaderBufferAccess(false), vk::AccessFlagBits2::eShaderRead);
}

TEST(BufferBarrierPolicy, WritableBindingPreservesShaderReadAccess) {
    EXPECT_EQ(Vulkan::ShaderBufferAccess(true),
              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
}

TEST(BufferBarrierPolicy, AliasedReadAndWriteBindingsMergeAccess) {
    auto access = Vulkan::ShaderBufferAccess(false);
    access = Vulkan::MergeShaderBufferAccess(access, true);

    EXPECT_EQ(access, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
}

TEST(BufferBarrierPolicy, ShaderAndIndirectUsagesShareOneDestinationBarrier) {
    std::array barriers{
        vk::BufferMemoryBarrier2{
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask =
                vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            .offset = 0,
            .size = 4096,
        },
    };
    const vk::BufferMemoryBarrier2 indirect_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask =
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eDrawIndirect,
        .dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead,
        .offset = 0,
        .size = 4096,
    };

    EXPECT_TRUE(Vulkan::TryMergeBufferBarrierDestination(barriers, indirect_barrier));
    EXPECT_EQ(barriers[0].dstStageMask, vk::PipelineStageFlagBits2::eAllCommands |
                                                vk::PipelineStageFlagBits2::eDrawIndirect);
    EXPECT_EQ(barriers[0].dstAccessMask,
              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
                  vk::AccessFlagBits2::eIndirectCommandRead);
    EXPECT_FALSE(barriers[0].srcAccessMask);
}

TEST(BufferBarrierPolicy, DifferentBufferRangesKeepIndependentBarriers) {
    std::array barriers{
        vk::BufferMemoryBarrier2{
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .offset = 0,
            .size = 4096,
        },
    };
    const vk::BufferMemoryBarrier2 other_range{
        .dstStageMask = vk::PipelineStageFlagBits2::eDrawIndirect,
        .dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead,
        .offset = 4096,
        .size = 4096,
    };

    EXPECT_FALSE(Vulkan::TryMergeBufferBarrierDestination(barriers, other_range));
    EXPECT_EQ(barriers[0].dstStageMask, vk::PipelineStageFlagBits2::eComputeShader);
    EXPECT_EQ(barriers[0].dstAccessMask, vk::AccessFlagBits2::eShaderWrite);
}

TEST(BufferBarrierPolicy, RepeatedShaderWritesStillRequireABarrier) {
    constexpr auto access =
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;

    EXPECT_TRUE(Vulkan::NeedsBufferBarrier(access, vk::PipelineStageFlagBits2::eAllCommands,
                                           access, vk::PipelineStageFlagBits2::eAllCommands));
}

TEST(BufferBarrierPolicy, RepeatedReadsDoNotRequireARedundantBarrier) {
    constexpr auto access = vk::AccessFlagBits2::eShaderRead;

    EXPECT_FALSE(Vulkan::NeedsBufferBarrier(access, vk::PipelineStageFlagBits2::eAllCommands,
                                            access, vk::PipelineStageFlagBits2::eAllCommands));
}

TEST(BufferBarrierPolicy, RepeatedTransferWritesStillRequireABarrier) {
    constexpr auto access = vk::AccessFlagBits2::eTransferWrite;

    EXPECT_TRUE(Vulkan::NeedsBufferBarrier(access, vk::PipelineStageFlagBits2::eTransfer, access,
                                           vk::PipelineStageFlagBits2::eTransfer));
}

TEST(BufferBarrierPolicy, AccessOrStageChangesRequireABarrier) {
    EXPECT_TRUE(Vulkan::NeedsBufferBarrier(
        vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eComputeShader));
    EXPECT_TRUE(Vulkan::NeedsBufferBarrier(
        vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eFragmentShader));
}
