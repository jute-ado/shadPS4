// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <sstream>
#include <gtest/gtest.h>

#include "video_core/texture_cache/image_subresource_layout.h"

namespace {
struct State {
    vk::ImageLayout layout;
};

TEST(ImageSubresourceLayout, DisjointMipBindingsUseTheirOwnLayouts) {
    const std::array states{State{vk::ImageLayout::eShaderReadOnlyOptimal},
                            State{vk::ImageLayout::eGeneral}};
    EXPECT_EQ(VideoCore::ResolveSubresourceLayout(vk::ImageLayout::eGeneral,
                                                  std::span<const State>{states}, {2, 1},
                                                  {{0, 0}, {1, 1}}),
              vk::ImageLayout::eShaderReadOnlyOptimal);
    EXPECT_EQ(VideoCore::ResolveSubresourceLayout(vk::ImageLayout::eShaderReadOnlyOptimal,
                                                  std::span<const State>{states}, {2, 1},
                                                  {{1, 0}, {1, 1}}),
              vk::ImageLayout::eGeneral);
}

TEST(ImageSubresourceLayout, MixedViewCannotPublishOneGuessedLayout) {
    const std::array states{State{vk::ImageLayout::eShaderReadOnlyOptimal},
                            State{vk::ImageLayout::eGeneral}};
    EXPECT_FALSE(VideoCore::ResolveSubresourceLayout(
        vk::ImageLayout::eGeneral, std::span<const State>{states}, {2, 1}, {{0, 0}, {2, 1}}));
}

TEST(ImageSubresourceLayout, UniformViewAndWholeImageFallbackRemainValid) {
    const std::array states{State{vk::ImageLayout::eGeneral}, State{vk::ImageLayout::eGeneral},
                            State{vk::ImageLayout::eShaderReadOnlyOptimal},
                            State{vk::ImageLayout::eShaderReadOnlyOptimal}};
    EXPECT_EQ(VideoCore::ResolveSubresourceLayout(vk::ImageLayout::eGeneral,
                                                  std::span<const State>{states}, {2, 2},
                                                  {{1, 0}, {1, 2}}),
              vk::ImageLayout::eShaderReadOnlyOptimal);
    EXPECT_EQ(VideoCore::ResolveSubresourceLayout(
                  vk::ImageLayout::eGeneral, std::span<const State>{}, {2, 2}, {{0, 0}, {2, 2}}),
              vk::ImageLayout::eGeneral);
}

TEST(ImageSubresourceLayout, InvalidRangesAndIncompleteStateFailClosed) {
    const std::array states{State{vk::ImageLayout::eGeneral}};
    for (const VideoCore::SubresourceRange range : {VideoCore::SubresourceRange{{2, 0}, {1, 1}},
                                                    {{0, 0}, {0, 1}},
                                                    {{0, 1}, {1, 1}},
                                                    {{0xffffffffU, 0}, {2, 1}}}) {
        EXPECT_FALSE(VideoCore::ResolveSubresourceLayout(vk::ImageLayout::eGeneral,
                                                         std::span<const State>{}, {2, 1}, range));
    }
    EXPECT_FALSE(VideoCore::ResolveSubresourceLayout(
        vk::ImageLayout::eGeneral, std::span<const State>{states}, {2, 1}, {{1, 0}, {1, 1}}));
}

TEST(ImageSubresourceLayout, DescriptorPublicationUsesTheViewRangeNotTheLastTransition) {
    std::ifstream input{SHADPS4_DESCRIPTOR_RASTERIZER_SOURCE_PATH};
    std::ostringstream stream;
    stream << input.rdbuf();
    const auto source = stream.str();
    const auto begin = source.find("bool Rasterizer::FinalizeImageDescriptors(");
    const auto end = source.find("void Rasterizer::TrackBoundImage(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_EQ(body.find("image->backing->state.layout"), std::string::npos);
    EXPECT_NE(body.find("GetSubresourceLayout(pending.view_info.range)"), std::string::npos);
}
} // namespace
