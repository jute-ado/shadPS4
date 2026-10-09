// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

#include "video_core/renderer_vulkan/deferred_resource_retirement.h"

TEST(DeferredResourceRetirement, PublishesOnlyAfterTheRecordingSubmissionGetsATick) {
    Vulkan::SubmissionResourceRetirements retirements;
    std::vector<std::pair<u64, int>> published;

    retirements.Retain([&published] { published.emplace_back(0, 7); });
    EXPECT_TRUE(published.empty());
    EXPECT_EQ(retirements.Size(), 1U);

    retirements.Publish(41, [&published](auto&& callback, u64 tick) {
        callback();
        published.back().first = tick;
    });

    EXPECT_EQ(published, (std::vector<std::pair<u64, int>>{{41, 7}}));
    EXPECT_EQ(retirements.Size(), 0U);
}

TEST(DeferredResourceRetirement, SchedulerPublishesAgainstTheSubmittedSignalValue) {
    std::ifstream header_input{SHADPS4_VK_SCHEDULER_HEADER_PATH, std::ios::binary};
    std::ostringstream header;
    header << header_input.rdbuf();
    EXPECT_NE(header.str().find("resource_retirements.Retain"), std::string::npos);

    std::ifstream source_input{SHADPS4_VK_SCHEDULER_SOURCE_PATH, std::ios::binary};
    std::ostringstream source;
    source << source_input.rdbuf();
    EXPECT_NE(source.str().find("resource_retirements.Publish(signal_value"), std::string::npos);
}

TEST(DeferredResourceRetirement, TileScratchIsRetainedOnlyAfterItsFinalRecordedUse) {
    std::ifstream tile_input{SHADPS4_TILE_MANAGER_SOURCE_PATH, std::ios::binary};
    std::ostringstream tile_stream;
    tile_stream << tile_input.rdbuf();
    const std::string tile_source = tile_stream.str();

    const auto tile_image = tile_source.find("void TileManager::TileImage");
    const auto tile_dispatch = tile_source.find("cmdbuf.dispatch", tile_image);
    const auto tile_retirement = tile_source.find("scheduler.DeferResourceDestruction", tile_image);
    EXPECT_NE(tile_image, std::string::npos);
    EXPECT_LT(tile_dispatch, tile_retirement);

    std::ifstream cache_input{SHADPS4_TEXTURE_CACHE_SOURCE_PATH, std::ios::binary};
    std::ostringstream cache_stream;
    cache_stream << cache_input.rdbuf();
    const std::string cache_source = cache_stream.str();
    const auto upload = cache_source.find("image.Upload(image_copies");
    const auto retirement = cache_source.find("tile_manager.RetireScratch", upload);
    EXPECT_NE(upload, std::string::npos);
    EXPECT_LT(upload, retirement);
}
