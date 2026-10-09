// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>
#include "video_core/buffer_cache/buffer_access_state.h"

namespace {
constexpr auto Read = vk::AccessFlagBits2::eShaderRead;
constexpr auto Write = vk::AccessFlagBits2::eShaderWrite;
constexpr auto Stage = vk::PipelineStageFlagBits2::eAllCommands;

TEST(BufferAccessState, RepeatedWriteNeedsMemoryDependency) {
    VideoCore::BufferAccessState state{Write, Stage};
    const auto barrier = state.GetBarrier(vk::Buffer{}, 4096, Write, Stage, 0);
    ASSERT_TRUE(barrier.has_value());
    EXPECT_TRUE(bool(barrier->srcAccessMask & Write));
    EXPECT_TRUE(bool(barrier->dstAccessMask & Write));
}

TEST(BufferAccessState, RepeatedReadWriteNeedsBothDestinationScopes) {
    VideoCore::BufferAccessState state{Read | Write, Stage};
    const auto barrier = state.GetBarrier(vk::Buffer{}, 4096, Read | Write, Stage, 0);
    ASSERT_TRUE(barrier.has_value());
    EXPECT_TRUE(bool(barrier->srcAccessMask & Write));
    EXPECT_EQ(barrier->dstAccessMask, Read | Write);
}

TEST(BufferAccessState, RepeatedReadOnlyRemainsElided) {
    VideoCore::BufferAccessState state{Read, Stage};
    EXPECT_FALSE(state.GetBarrier(vk::Buffer{}, 4096, Read, Stage, 0));
}

TEST(BufferAccessState, WriteToReadControlPreservesRangeAndMasks) {
    VideoCore::BufferAccessState state{Write, Stage};
    const auto barrier = state.GetBarrier(vk::Buffer{}, 4096, Read, Stage, 64);
    ASSERT_TRUE(barrier.has_value());
    EXPECT_EQ(barrier->srcAccessMask, Write);
    EXPECT_EQ(barrier->dstAccessMask, Read);
    EXPECT_EQ(barrier->offset, 64);
    EXPECT_EQ(barrier->size, 4032);
    EXPECT_EQ(state.access_mask, Read);
    EXPECT_EQ(state.stage, Stage);
    EXPECT_FALSE(state.GetBarrier(vk::Buffer{}, 4096, Read, Stage, 64));
}

TEST(BufferAccessState, WritableShaderDescriptorRetainsReadAccess) {
    EXPECT_EQ(VideoCore::ShaderBufferAccess(true), Read | Write);
    EXPECT_EQ(VideoCore::ShaderBufferAccess(false), Read);
}

TEST(BufferAccessState, RepeatedTransferAndGenericWritesNeedDependencies) {
    for (const auto access : {vk::AccessFlagBits2::eTransferWrite,
                              vk::AccessFlagBits2::eMemoryWrite,
                              vk::AccessFlagBits2::eHostWrite}) {
        VideoCore::BufferAccessState state{access, Stage};
        EXPECT_TRUE(state.GetBarrier(vk::Buffer{}, 4096, access, Stage, 0));
    }
}

TEST(BufferAccessState, ReadOnlyAccessCombinationsRemainElided) {
    const vk::AccessFlags2 reads = vk::AccessFlagBits2::eIndirectCommandRead |
                                  vk::AccessFlagBits2::eIndexRead |
                                  vk::AccessFlagBits2::eVertexAttributeRead |
                                  vk::AccessFlagBits2::eUniformRead | Read |
                                  vk::AccessFlagBits2::eTransferRead |
                                  vk::AccessFlagBits2::eHostRead |
                                  vk::AccessFlagBits2::eMemoryRead;
    VideoCore::BufferAccessState state{reads, Stage};
    EXPECT_FALSE(state.GetBarrier(vk::Buffer{}, 4096, reads, Stage));
}

TEST(BufferAccessState, UnknownAccessBitCannotProveReadOnly) {
    const vk::AccessFlags2 unknown{VkAccessFlags2{1} << 63};
    VideoCore::BufferAccessState state{unknown, Stage};
    EXPECT_TRUE(state.GetBarrier(vk::Buffer{}, 4096, unknown, Stage));
}

TEST(BufferAccessState, ChangedReadStageRetainsExecutionDependency) {
    VideoCore::BufferAccessState state{Read, vk::PipelineStageFlagBits2::eComputeShader};
    const auto barrier = state.GetBarrier(vk::Buffer{}, 4096, Read, Stage);
    ASSERT_TRUE(barrier.has_value());
    EXPECT_EQ(barrier->srcStageMask, vk::PipelineStageFlagBits2::eComputeShader);
    EXPECT_EQ(barrier->dstStageMask, Stage);
    EXPECT_EQ(state.access_mask, Read);
    EXPECT_EQ(state.stage, Stage);
    EXPECT_FALSE(state.GetBarrier(vk::Buffer{}, 4096, Read, Stage));
}

TEST(BufferAccessState, AggregatedAliasesRetainEveryAccessAndPipelineStage) {
    constexpr auto VertexRead = vk::AccessFlagBits2::eVertexAttributeRead;
    constexpr auto IndexRead = vk::AccessFlagBits2::eIndexRead;
    constexpr auto ShaderWrite = vk::AccessFlagBits2::eShaderWrite;
    const vk::AccessFlags2 combined_access = VertexRead | IndexRead | ShaderWrite;
    const vk::PipelineStageFlags2 combined_stages =
        vk::PipelineStageFlagBits2::eVertexAttributeInput |
        vk::PipelineStageFlagBits2::eIndexInput | vk::PipelineStageFlagBits2::eComputeShader;

    VideoCore::BufferAccessState state{Write, Stage};
    const auto barrier =
        state.GetBarrier(vk::Buffer{}, 4096, combined_access, combined_stages);

    ASSERT_TRUE(barrier.has_value());
    EXPECT_EQ(barrier->dstAccessMask, combined_access);
    EXPECT_EQ(barrier->dstStageMask, combined_stages);
    EXPECT_EQ(state.access_mask, combined_access);
    EXPECT_EQ(state.stage, combined_stages);
}

TEST(BufferAccessState, ProductionUsesTestedStateAndDescriptorClassification) {
    // Keep this source contract aligned with the retained atomic-batch design; a restored
    // checkout must not inherit the later rejected forced-canonical expectations.
    const auto read = [](const char* suffix) {
        std::ifstream file(std::string(BUFFER_ACCESS_SOURCE_ROOT) + suffix);
        return std::string(std::istreambuf_iterator<char>{file}, {});
    };
    const auto buffer = read("/src/video_core/buffer_cache/buffer.h");
    const auto buffer_cache = read("/src/video_core/buffer_cache/buffer_cache.cpp");
    const auto rasterizer = read("/src/video_core/renderer_vulkan/vk_rasterizer.cpp");
    ASSERT_FALSE(buffer.empty());
    ASSERT_FALSE(buffer_cache.empty());
    ASSERT_FALSE(rasterizer.empty());
    EXPECT_NE(buffer.find("BufferAccessState state{access_mask, stage}"), std::string::npos);
    EXPECT_NE(buffer.find("state.GetBarrier(buffer.buffer, size_bytes"), std::string::npos);
    EXPECT_NE(buffer.find("access_mask = state.access_mask"), std::string::npos);
    EXPECT_NE(buffer.find("stage = state.stage"), std::string::npos);
    EXPECT_NE(rasterizer.find(".access = VideoCore::ShaderBufferAccess(desc.is_written)"),
              std::string::npos);
    EXPECT_NE(rasterizer.find("pending.prepared, pending.access"), std::string::npos);
    EXPECT_NE(buffer_cache.find("existing->access_mask |= access"), std::string::npos);
    EXPECT_NE(buffer_cache.find("existing->stage_mask |= stage"), std::string::npos);
    EXPECT_NE(buffer_cache.find("access.buffer->GetBarrier(access.access_mask, access.stage_mask)"),
              std::string::npos);
}
} // namespace
