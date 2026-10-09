// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "video_core/quad_list_indices.h"

TEST(QuadListIndices, CountsOnlyCompleteQuadsWithoutOverflow) {
    EXPECT_EQ(VideoCore::QuadListTriangleIndexCount(0), 0u);
    EXPECT_EQ(VideoCore::QuadListTriangleIndexCount(3), 0u);
    EXPECT_EQ(VideoCore::QuadListTriangleIndexCount(4), 6u);
    EXPECT_EQ(VideoCore::QuadListTriangleIndexCount(9), 12u);
    EXPECT_EQ(VideoCore::QuadListTriangleIndexCount(std::numeric_limits<u32>::max()), 0u);
}

TEST(QuadListIndices, GeneratesTriangleIndicesForDirectNonIndexedDraws) {
    std::array<u32, 12> output{};
    VideoCore::GenerateQuadListTriangleIndices(output, 8);
    EXPECT_EQ(output, (std::array<u32, 12>{0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7}));
}

TEST(QuadListIndices, ConvertsSixteenBitGuestIndices) {
    constexpr std::array<u16, 4> input{9, 4, 7, 2};
    std::array<u16, 6> output{};
    VideoCore::ConvertQuadListTriangleIndices(output, input);
    EXPECT_EQ(output, (std::array<u16, 6>{9, 4, 7, 9, 7, 2}));
}

TEST(QuadListIndices, ConvertsThirtyTwoBitGuestIndicesAndIgnoresIncompleteTail) {
    constexpr std::array<u32, 6> input{100000, 8, 42, 77, 91, 92};
    std::array<u32, 6> output{};
    VideoCore::ConvertQuadListTriangleIndices(output, input);
    EXPECT_EQ(output, (std::array<u32, 6>{100000, 8, 42, 100000, 42, 77}));
}

TEST(QuadListIndices, LeavesTooSmallOutputsUntouched) {
    constexpr std::array<u32, 4> input{1, 2, 3, 4};
    std::array<u32, 5> output{9, 9, 9, 9, 9};
    VideoCore::ConvertQuadListTriangleIndices(output, input);
    EXPECT_EQ(output, (std::array<u32, 5>{9, 9, 9, 9, 9}));
}

TEST(QuadListIndices, ConvertsOnlyPlainVertexQuadListDrawsWithoutPrimitiveRestart) {
    using AmdGpu::PrimitiveType;
    using VgtStages = AmdGpu::ShaderStageEnable::VgtStages;

    EXPECT_TRUE(VideoCore::ShouldConvertQuadListToTriangles(PrimitiveType::QuadList, VgtStages::Vs,
                                                            false));
    EXPECT_FALSE(VideoCore::ShouldConvertQuadListToTriangles(
        PrimitiveType::QuadList, VgtStages::Vs, true));
    EXPECT_FALSE(VideoCore::ShouldConvertQuadListToTriangles(
        PrimitiveType::QuadList, VgtStages::EsGs, false));
    EXPECT_FALSE(VideoCore::ShouldConvertQuadListToTriangles(
        PrimitiveType::QuadList, VgtStages::LsHs, false));
    EXPECT_FALSE(VideoCore::ShouldConvertQuadListToTriangles(
        PrimitiveType::QuadList, VgtStages::LsHsEsGs, false));
    EXPECT_FALSE(VideoCore::ShouldConvertQuadListToTriangles(
        PrimitiveType::TriangleList, VgtStages::Vs, false));
}

TEST(QuadListIndices, PlansOneEffectiveIndexedTriangleDrawWithoutChangingOtherDraws) {
    using AmdGpu::PrimitiveType;
    using VgtStages = AmdGpu::ShaderStageEnable::VgtStages;

    const auto direct = VideoCore::PlanQuadListDraw(PrimitiveType::QuadList, VgtStages::Vs, false,
                                                    false, 1260);
    EXPECT_TRUE(direct.convert);
    EXPECT_TRUE(direct.indexed);
    EXPECT_EQ(direct.primitive_type, PrimitiveType::TriangleList);
    EXPECT_EQ(direct.index_count, 1890u);

    const auto guest_indexed = VideoCore::PlanQuadListDraw(
        PrimitiveType::QuadList, VgtStages::Vs, false, true, 8);
    EXPECT_TRUE(guest_indexed.convert);
    EXPECT_TRUE(guest_indexed.indexed);
    EXPECT_EQ(guest_indexed.primitive_type, PrimitiveType::TriangleList);
    EXPECT_EQ(guest_indexed.index_count, 12u);

    const auto unchanged = VideoCore::PlanQuadListDraw(
        PrimitiveType::TriangleStrip, VgtStages::Vs, false, false, 7);
    EXPECT_FALSE(unchanged.convert);
    EXPECT_FALSE(unchanged.indexed);
    EXPECT_EQ(unchanged.primitive_type, PrimitiveType::TriangleStrip);
    EXPECT_EQ(unchanged.index_count, 7u);
}

TEST(QuadListIndices, RuntimeUsesThePlanForPipelineBindingAndDrawCount) {
    const auto read_source = [](const char* path) {
        std::ifstream stream{path};
        std::ostringstream text;
        text << stream.rdbuf();
        return text.str();
    };
    const std::string rasterizer = read_source(QUAD_LIST_RASTERIZER_SOURCE_PATH);
    const std::string pipeline_cache = read_source(QUAD_LIST_PIPELINE_CACHE_SOURCE_PATH);

    const auto draw_begin = rasterizer.find("void Rasterizer::Draw(bool is_indexed");
    const auto draw_end = rasterizer.find("void Rasterizer::DrawIndirect", draw_begin);
    ASSERT_NE(draw_begin, std::string::npos);
    ASSERT_NE(draw_end, std::string::npos);
    const std::string draw = rasterizer.substr(draw_begin, draw_end - draw_begin);

    EXPECT_NE(draw.find("PlanQuadListDraw("), std::string::npos);
    EXPECT_NE(draw.find("GetGraphicsPipeline(quad_list_plan.primitive_type)"), std::string::npos);
    EXPECT_NE(draw.find("BindQuadListIndexBuffer(is_indexed, index_offset)"),
              std::string::npos);
    EXPECT_NE(draw.find("drawIndexed(index_count"), std::string::npos);
    EXPECT_NE(draw.find("UpdateDynamicState(pipeline, is_indexed)"), std::string::npos);

    EXPECT_NE(pipeline_cache.find("PipelineCache::GetGraphicsPipeline("),
              std::string::npos);
    EXPECT_NE(pipeline_cache.find("AmdGpu::PrimitiveType primitive_type)"), std::string::npos);
    EXPECT_NE(pipeline_cache.find("key.prim_type = primitive_type;"), std::string::npos);
    EXPECT_NE(pipeline_cache.find("graphics_key.prim_type == AmdGpu::PrimitiveType::QuadList"),
              std::string::npos);
}
