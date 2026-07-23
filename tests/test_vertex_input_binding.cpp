// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/vertex_buffer_range.h"
#include "video_core/buffer_cache/vertex_input_binding.h"

namespace {

struct TestBuffer {
    VAddr base_address;
    u64 size;

    u64 GetSize() const {
        return size;
    }
};

} // namespace

TEST(VertexBufferRange, ClampsDescriptorsBeforeMergingThem) {
    constexpr std::array buffers{
        TestBuffer{.base_address = 0x1000, .size = 0x3ffff'fffc},
        TestBuffer{.base_address = 0x5000, .size = 0x3ffff'fffc},
    };

    const auto ranges = VideoCore::BuildVertexBufferRanges(
        buffers, [](VAddr address, u64) { return address == 0x1000 ? u64{0x1000} : u64{0x800}; });

    ASSERT_EQ(ranges.size(), 2u);
    EXPECT_EQ(ranges[0].base_address, 0x1000u);
    EXPECT_EQ(ranges[0].end_address, 0x2000u);
    EXPECT_EQ(ranges[1].base_address, 0x5000u);
    EXPECT_EQ(ranges[1].end_address, 0x5800u);
}

TEST(VertexBufferRange, MergesOverlappingClampedRanges) {
    constexpr std::array buffers{
        TestBuffer{.base_address = 0x1000, .size = 0x400},
        TestBuffer{.base_address = 0x1200, .size = 0x400},
    };

    const auto ranges =
        VideoCore::BuildVertexBufferRanges(buffers, [](VAddr, u64 size) { return size; });

    ASSERT_EQ(ranges.size(), 1u);
    EXPECT_EQ(ranges[0].base_address, 0x1000u);
    EXPECT_EQ(ranges[0].end_address, 0x1600u);
}

TEST(VertexInputBinding, DrawCommitsDynamicStateAfterPipelineBinding) {
    std::vector<std::string_view> operations;

    VideoCore::IssueDrawWithVertexInputState(
        true, [&] { operations.emplace_back("pipeline"); },
        [&] { operations.emplace_back("dynamic"); }, [&] { operations.emplace_back("draw"); });

    EXPECT_EQ(operations, (std::vector<std::string_view>{"pipeline", "dynamic", "draw"}));
}

TEST(VertexInputBinding, StaticDrawSkipsDynamicState) {
    std::vector<std::string_view> operations;

    VideoCore::IssueDrawWithVertexInputState(
        false, [&] { operations.emplace_back("pipeline"); },
        [&] { operations.emplace_back("dynamic"); }, [&] { operations.emplace_back("draw"); });

    EXPECT_EQ(operations, (std::vector<std::string_view>{"pipeline", "draw"}));
}
