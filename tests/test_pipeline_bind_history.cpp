// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_pipeline_bind_history.h"

using Vulkan::PipelineBindHistory;
using Vulkan::PipelineBindRecord;
using Vulkan::PipelineBindType;

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
