// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/vertex_input_binding.h"

TEST(VertexInputBinding, DrawCommitsDynamicStateAfterPipelineBinding) {
    std::vector<std::string_view> operations;

    VideoCore::IssueDrawWithVertexInputState(
        true, [&] { operations.emplace_back("pipeline"); },
        [&] { operations.emplace_back("dynamic"); }, [&] { operations.emplace_back("draw"); });

    EXPECT_EQ(operations,
              (std::vector<std::string_view>{"pipeline", "dynamic", "draw"}));
}

TEST(VertexInputBinding, StaticDrawSkipsDynamicState) {
    std::vector<std::string_view> operations;

    VideoCore::IssueDrawWithVertexInputState(
        false, [&] { operations.emplace_back("pipeline"); },
        [&] { operations.emplace_back("dynamic"); }, [&] { operations.emplace_back("draw"); });

    EXPECT_EQ(operations, (std::vector<std::string_view>{"pipeline", "draw"}));
}
