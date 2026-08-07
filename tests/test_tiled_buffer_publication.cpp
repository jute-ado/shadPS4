// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/texture_cache/tiled_buffer_publication.h"

namespace VideoCore {
namespace {

TEST(TiledBufferPublication, AcquiresAndPublishesComputeWritesBeforeShaderReads) {
    const BufferAccessState previous{
        .stage = vk::PipelineStageFlagBits2::eAllCommands,
        .access = vk::AccessFlagBits2::eShaderRead,
    };

    const auto plan = GetTiledBufferPublicationPlan(previous, true);

    ASSERT_TRUE(plan.required);
    EXPECT_EQ(plan.before_write.src, previous);
    EXPECT_EQ(plan.before_write.dst.stage, vk::PipelineStageFlagBits2::eComputeShader);
    EXPECT_EQ(plan.before_write.dst.access, vk::AccessFlagBits2::eShaderWrite);
    EXPECT_EQ(plan.before_read.src, plan.before_write.dst);
    EXPECT_EQ(plan.before_read.dst.stage, vk::PipelineStageFlagBits2::eAllCommands);
    EXPECT_EQ(plan.before_read.dst.access, vk::AccessFlagBits2::eShaderRead);
}

TEST(TiledBufferPublication, OrdersRepeatedComputeWritesEvenWhenAccessStateMatches) {
    const BufferAccessState previous{
        .stage = vk::PipelineStageFlagBits2::eComputeShader,
        .access = vk::AccessFlagBits2::eShaderWrite,
    };

    const auto plan = GetTiledBufferPublicationPlan(previous, true);

    ASSERT_TRUE(plan.required);
    EXPECT_TRUE(plan.before_write.required);
    EXPECT_EQ(plan.before_write.src, plan.before_write.dst);
    EXPECT_TRUE(plan.before_read.required);
}

TEST(TiledBufferPublication, LeavesLinearImageDownloadOnItsTransferPublicationPath) {
    const BufferAccessState previous{
        .stage = vk::PipelineStageFlagBits2::eAllCommands,
        .access = vk::AccessFlagBits2::eShaderRead,
    };

    const auto plan = GetTiledBufferPublicationPlan(previous, false);

    EXPECT_FALSE(plan.required);
}

} // namespace
} // namespace VideoCore
