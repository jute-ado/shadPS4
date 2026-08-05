// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdint>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/buffer_access_batch.h"

namespace {

using Batch = VideoCore::BasicBufferAccessBatch<std::uint32_t>;

TEST(BufferAccessBatch, AliasedReadWriteIsOrderIndependent) {
    Batch write_then_read;
    write_then_read.Add(7, vk::AccessFlagBits2::eShaderWrite,
                        vk::PipelineStageFlagBits2::eComputeShader);
    write_then_read.Add(7, vk::AccessFlagBits2::eShaderRead,
                        vk::PipelineStageFlagBits2::eVertexShader);

    Batch read_then_write;
    read_then_write.Add(7, vk::AccessFlagBits2::eShaderRead,
                        vk::PipelineStageFlagBits2::eVertexShader);
    read_then_write.Add(7, vk::AccessFlagBits2::eShaderWrite,
                        vk::PipelineStageFlagBits2::eComputeShader);

    ASSERT_EQ(write_then_read.Entries().size(), 1);
    ASSERT_EQ(read_then_write.Entries().size(), 1);
    EXPECT_EQ(write_then_read.Entries()[0], read_then_write.Entries()[0]);
    EXPECT_EQ(write_then_read.Entries()[0].access,
              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    EXPECT_EQ(write_then_read.Entries()[0].stages, vk::PipelineStageFlagBits2::eComputeShader |
                                                       vk::PipelineStageFlagBits2::eVertexShader);
}

TEST(BufferAccessBatch, WritableBindingIncludesReadAndWrite) {
    EXPECT_EQ(VideoCore::ShaderBufferAccess(false), vk::AccessFlagBits2::eShaderRead);
    EXPECT_EQ(VideoCore::ShaderBufferAccess(true),
              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
}

TEST(BufferAccessBatch, StorageWriteVertexAndIndirectReadCommitOneUnion) {
    Batch batch;
    batch.Add(19, VideoCore::ShaderBufferAccess(true), vk::PipelineStageFlagBits2::eComputeShader);
    batch.Add(19, vk::AccessFlagBits2::eVertexAttributeRead,
              vk::PipelineStageFlagBits2::eVertexAttributeInput);
    batch.Add(19, vk::AccessFlagBits2::eIndirectCommandRead,
              vk::PipelineStageFlagBits2::eDrawIndirect);

    ASSERT_EQ(batch.Entries().size(), 1);
    const auto& entry = batch.Entries()[0];
    EXPECT_EQ(entry.access, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
                                vk::AccessFlagBits2::eVertexAttributeRead |
                                vk::AccessFlagBits2::eIndirectCommandRead);
    EXPECT_EQ(entry.stages, vk::PipelineStageFlagBits2::eComputeShader |
                                vk::PipelineStageFlagBits2::eVertexAttributeInput |
                                vk::PipelineStageFlagBits2::eDrawIndirect);
}

TEST(BufferAccessBatch, NextReadDependsOnPriorAliasedWrite) {
    Batch first_command;
    first_command.Add(23, vk::AccessFlagBits2::eShaderWrite,
                      vk::PipelineStageFlagBits2::eComputeShader);
    first_command.Add(23, vk::AccessFlagBits2::eShaderRead,
                      vk::PipelineStageFlagBits2::eVertexShader);

    const auto& prior = first_command.Entries()[0];
    EXPECT_TRUE(VideoCore::NeedsBufferBarrier(prior.access, prior.stages,
                                              vk::AccessFlagBits2::eShaderRead,
                                              vk::PipelineStageFlagBits2::eVertexShader));
}

TEST(BufferAccessBatch, IdenticalReadOnlyAccessCanReuseDependency) {
    EXPECT_FALSE(VideoCore::NeedsBufferBarrier(
        vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eVertexShader,
        vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eVertexShader));
}

TEST(BufferAccessBatch, IdenticalWritesRemainDependent) {
    EXPECT_TRUE(VideoCore::NeedsBufferBarrier(
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eComputeShader));
}

TEST(BufferAccessBatch, ObservationsDoNotCreateSyntheticSequentialTransitions) {
    Batch batch;
    batch.Add(31, vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader);
    batch.Add(31, vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eVertexShader);
    batch.Add(31, vk::AccessFlagBits2::eIndexRead, vk::PipelineStageFlagBits2::eIndexInput);

    ASSERT_EQ(batch.Entries().size(), 1);
    EXPECT_EQ(batch.Entries()[0].key, 31);
}

TEST(BufferAccessBatch, ClearStartsANewCommand) {
    Batch batch;
    batch.Add(37, vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader);
    batch.Clear();
    batch.Add(37, vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eVertexShader);

    ASSERT_EQ(batch.Entries().size(), 1);
    EXPECT_EQ(batch.Entries()[0].access, vk::AccessFlagBits2::eShaderRead);
    EXPECT_EQ(batch.Entries()[0].stages, vk::PipelineStageFlagBits2::eVertexShader);
}

} // namespace
