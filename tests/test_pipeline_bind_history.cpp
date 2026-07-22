// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_pipeline_bind_history.h"

using Vulkan::PipelineBindHistory;
using Vulkan::PipelineBindRecord;
using Vulkan::PipelineBindType;
using Vulkan::PipelineBufferInfo;
using Vulkan::PipelineCommandInfo;
using Vulkan::PipelineCommandType;
using Vulkan::PipelineImageInfo;

TEST(PipelineBindHistory, RetainsDispatchKindAndDimensions) {
    constexpr std::array program_hashes{0x11ull};
    constexpr PipelineCommandInfo command{
        .type = PipelineCommandType::Dispatch,
        .arguments = {64, 32, 1},
    };

    const auto record =
        Vulkan::MakePipelineBindRecord(PipelineBindType::Compute, 0x55, program_hashes, command);

    EXPECT_EQ(record.command, command);
}

TEST(PipelineBindHistory, RetainsBoundBufferAddressSizesAndAccess) {
    constexpr std::array program_hashes{0x11ull};
    constexpr std::array buffers{
        PipelineBufferInfo{
            .base_address = 0x119eda8d00,
            .requested_size = 0x3fffffffc,
            .bound_size = 0x1be57300,
            .stride = 4,
            .num_records = 0xffffffff,
            .is_written = false,
            .is_formatted = false,
        },
        PipelineBufferInfo{
            .base_address = 0x11005e8000,
            .requested_size = 0x4000,
            .bound_size = 0x4000,
            .stride = 16,
            .num_records = 0x400,
            .is_written = true,
            .is_formatted = true,
        },
    };

    const auto record = Vulkan::MakePipelineBindRecord(
        PipelineBindType::Compute, 0x55, program_hashes, {}, buffers);

    EXPECT_EQ(record.buffer_count, buffers.size());
    EXPECT_EQ(record.buffers[0], buffers[0]);
    EXPECT_EQ(record.buffers[1], buffers[1]);
}

TEST(PipelineBindHistory, RetainsBoundImageDimensionsFormatAndAccess) {
    constexpr std::array program_hashes{0x11ull};
    constexpr std::array images{
        PipelineImageInfo{
            .base_address = 0x119de0ec00,
            .width = 128,
            .height = 128,
            .depth = 1,
            .pitch = 128,
            .data_format = 10,
            .type = 9,
            .is_written = true,
        },
    };

    const auto record = Vulkan::MakePipelineBindRecord(
        PipelineBindType::Compute, 0x55, program_hashes, {}, {}, images);

    EXPECT_EQ(record.image_count, images.size());
    EXPECT_EQ(record.images[0], images[0]);
}

TEST(PipelineBindHistory, MakesRecordFromProgramHashesAndPadsUnusedStages) {
    constexpr std::array program_hashes{0x11ull, 0x22ull, 0ull, 0x44ull};

    const auto record =
        Vulkan::MakePipelineBindRecord(PipelineBindType::Graphics, 0x55, program_hashes);

    EXPECT_EQ(record.type, PipelineBindType::Graphics);
    EXPECT_EQ(record.pipeline_hash, 0x55u);
    EXPECT_EQ(record.shader_hashes,
              (std::array<u64, PipelineBindRecord::MaxShaderHashes>{0x11, 0x22, 0, 0x44, 0, 0}));
}

TEST(PipelineBindHistory, ReturnsUniquePipelinesNewestFirst) {
    PipelineBindHistory history;
    const PipelineBindRecord graphics{
        .type = PipelineBindType::Graphics,
        .pipeline_hash = 0x10,
        .shader_hashes = {0x20, 0x30},
    };
    const PipelineBindRecord compute{
        .type = PipelineBindType::Compute,
        .pipeline_hash = 0x40,
        .shader_hashes = {0x50},
    };

    history.Record(graphics);
    history.Record(compute);
    history.Record(graphics);

    const auto recent = history.RecentUnique();
    ASSERT_EQ(recent.size(), 2u);
    EXPECT_EQ(recent[0], graphics);
    EXPECT_EQ(recent[1], compute);
}

TEST(PipelineBindHistory, RetainsOnlyTheNewestFixedCapacityWindow) {
    PipelineBindHistory history;
    for (u64 hash = 1; hash <= PipelineBindHistory::Capacity + 3; ++hash) {
        history.Record(PipelineBindRecord{
            .type = PipelineBindType::Compute,
            .pipeline_hash = hash,
            .shader_hashes = {hash + 100},
        });
    }

    const auto recent = history.RecentUnique();
    ASSERT_EQ(recent.size(), PipelineBindHistory::Capacity);
    EXPECT_EQ(recent.front().pipeline_hash, PipelineBindHistory::Capacity + 3);
    EXPECT_EQ(recent.back().pipeline_hash, 4u);
}
