// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "video_core/amdgpu/index_draw_state.h"

namespace {

// Only field storage is substituted. Every mutation and preparation query below
// uses the same production helper as Liverpool/BufferCache, not a second model.
struct RegisterFields {
    std::uint32_t index_size{};
    std::uint32_t max_index_size{};
    std::uint32_t num_indices{};
};

// AMD SI guide sections2.5.1/2.5.10: INDEX_BUFFER_SIZE supplies CP capacity;
// direct/auto draw count updates must not destroy that independent input.
TEST(IndexDrawCapacity, AutoDrawDoesNotReplaceRetainedIndirectCapacity) {
    RegisterFields regs{};
    AmdGpu::ApplyIndexBufferSize(regs, 65536);
    AmdGpu::ApplyAutoIndexDrawSize(regs, 3);
    EXPECT_EQ(regs.num_indices, 3U);
    EXPECT_EQ(AmdGpu::CurrentIndexPreparationCount(regs), 65536U);
}

TEST(IndexDrawCapacity, DirectMaximumIsNotRetainedIndirectCapacity) {
    RegisterFields regs{};
    AmdGpu::ApplyIndexBufferSize(regs, 65536);
    AmdGpu::ApplyDirectIndexDrawSize(regs, 12, 3);
    EXPECT_EQ(regs.max_index_size, 12U);
    EXPECT_EQ(regs.num_indices, 3U);
    // Neither12 nor3 is the earlier CP INDEX_BUFFER_SIZE value.
    EXPECT_EQ(AmdGpu::CurrentIndexPreparationCount(regs), 65536U);
}

TEST(IndexDrawCapacity, IndexBufferSizeDoesNotWriteTheDirectDrawCountRegister) {
    RegisterFields regs{.max_index_size = 17, .num_indices = 7};
    AmdGpu::ApplyIndexBufferSize(regs, 65536);
    EXPECT_EQ(regs.num_indices, 7U);
    EXPECT_EQ(regs.max_index_size, 17U);
}

TEST(IndexDrawCapacity, ValidU32CapacityMustNotSilentlyWrapItsByteSpan) {
    RegisterFields regs{};
    AmdGpu::ApplyIndexBufferSize(regs, 0x40000000U);
    // A future admission policy may reject this large span instead; silently
    // acquiring an empty buffer is neither the wide size nor an explicit refusal.
    EXPECT_EQ(std::uint64_t{AmdGpu::CurrentIndexPreparationBytes(regs, 4)}, 0x100000000ULL);
}

TEST(IndexDrawCapacity, DirectPreparationUsesTheIndependentDrawCount) {
    RegisterFields regs{};
    AmdGpu::ApplyIndexBufferSize(regs, 65536);
    AmdGpu::ApplyAutoIndexDrawSize(regs, 3);
    const auto bytes = AmdGpu::TryGetIndexPreparationBytes(regs, 4, false);
    ASSERT_TRUE(bytes.has_value());
    EXPECT_EQ(*bytes, 12U);
}

TEST(IndexDrawCapacity, IndirectPreparationRejectsUnrepresentableWideSpan) {
    RegisterFields regs{};
    AmdGpu::ApplyIndexBufferSize(regs, 0x40000000U);
    EXPECT_FALSE(AmdGpu::TryGetIndexPreparationBytes(regs, 4, true).has_value());
}

TEST(IndexDrawCapacity, IndexAddressOffsetUsesWidenedCheckedArithmetic) {
    EXPECT_EQ(AmdGpu::TryGetIndexAddress(0x200000, 3, 4), 0x20000cULL);
    EXPECT_FALSE(AmdGpu::TryGetIndexAddress(std::numeric_limits<std::uint64_t>::max() - 1,
                                            1, 4)
                     .has_value());
}

TEST(IndexDrawCapacity, IndirectBufferRangeRejectsAddressAndSizeOverflow) {
    const auto valid = AmdGpu::TryGetIndirectBufferRange(0x200000, 0x100, 0x200);
    ASSERT_TRUE(valid.has_value());
    EXPECT_EQ(valid->address, 0x200100ULL);
    EXPECT_EQ(valid->size, 0x200U);
    EXPECT_FALSE(AmdGpu::TryGetIndirectBufferRange(std::numeric_limits<std::uint64_t>::max(), 1,
                                                   4)
                     .has_value());
    EXPECT_FALSE(AmdGpu::TryGetIndirectBufferRange(0x200000, 0, 0x100000000ULL).has_value());
}

TEST(IndexDrawCapacityControl, ExplicitCapacityImmediatelyBeforeIndirectIsObserved) {
    RegisterFields regs{};
    AmdGpu::ApplyAutoIndexDrawSize(regs, 3);
    AmdGpu::ApplyIndexBufferSize(regs, 65536);
    EXPECT_EQ(AmdGpu::CurrentIndexPreparationCount(regs), 65536U);
    EXPECT_EQ(AmdGpu::CurrentIndexPreparationBytes(regs, 4), 262144U);
}

TEST(IndexDrawCapacityControl, LaterExplicitCapacityReplacesEarlierCapacity) {
    RegisterFields regs{};
    AmdGpu::ApplyIndexBufferSize(regs, 65536);
    AmdGpu::ApplyIndexBufferSize(regs, 42);
    EXPECT_EQ(AmdGpu::CurrentIndexPreparationCount(regs), 42U);
}

TEST(IndexDrawCapacityControl, ExplicitZeroDoesNotReusePreviousNonzeroSize) {
    RegisterFields regs{};
    AmdGpu::ApplyIndexBufferSize(regs, 65536);
    AmdGpu::ApplyIndexBufferSize(regs, 0);
    EXPECT_EQ(AmdGpu::CurrentIndexPreparationBytes(regs, 4), 0U);
}

TEST(IndexDrawCapacityControl, DirectDrawStillUpdatesBothDirectFields) {
    RegisterFields regs{.max_index_size = 17, .num_indices = 7};
    AmdGpu::ApplyDirectIndexDrawSize(regs, 128, 24);
    EXPECT_EQ(regs.max_index_size, 128U);
    EXPECT_EQ(regs.num_indices, 24U);
}

TEST(IndexDrawCapacityControl, AutoDrawPreservesTheUnrelatedDirectMaximum) {
    RegisterFields regs{.max_index_size = 17, .num_indices = 7};
    AmdGpu::ApplyAutoIndexDrawSize(regs, 3);
    EXPECT_EQ(regs.max_index_size, 17U);
    EXPECT_EQ(regs.num_indices, 3U);
}

TEST(IndexDrawCapacityControl, IndexWidthsAndLargestNonoverflowingSpansRemainExact) {
    RegisterFields regs{};
    AmdGpu::ApplyIndexBufferSize(regs, 0x3fffffffU);
    EXPECT_EQ(AmdGpu::CurrentIndexPreparationBytes(regs, 4), 0xfffffffcU);
    AmdGpu::ApplyIndexBufferSize(regs, 0x7fffffffU);
    EXPECT_EQ(AmdGpu::CurrentIndexPreparationBytes(regs, 2), 0xfffffffeU);
}

std::string Source(const char* path) {
    std::ifstream file{path, std::ios::binary};
    EXPECT_TRUE(file.is_open());
    std::string text{std::istreambuf_iterator<char>{file}, {}};
    std::erase(text, '\r');
    return text;
}

TEST(IndexDrawCapacityWiring, ActualPacketHandlersUseTheCharacterizedTransitions) {
    const auto text = Source(SHADPS4_INDEX_DRAW_LIVERPOOL_SOURCE);
    const auto require = [&](std::string_view opcode, std::string_view operation) {
        const auto begin = text.find(opcode);
        ASSERT_NE(begin, std::string::npos);
        const auto end = text.find("case PM4ItOpcode::", begin + opcode.size());
        ASSERT_NE(end, std::string::npos);
        EXPECT_NE(text.substr(begin, end - begin).find(operation), std::string::npos);
    };
    require("case PM4ItOpcode::IndexBufferSize:", "ApplyIndexBufferSize(regs,");
    require("case PM4ItOpcode::DrawIndex2:", "ApplyDirectIndexDrawSize(regs,");
    require("case PM4ItOpcode::DrawIndexOffset2:", "ApplyDirectIndexDrawSize(regs,");
    require("case PM4ItOpcode::DrawIndexAuto:", "ApplyAutoIndexDrawSize(regs,");
}

TEST(IndexDrawCapacityWiring, ActualPreparationUsesTheSameCountAndU32Arithmetic) {
    const auto text = Source(SHADPS4_INDEX_DRAW_BUFFER_CACHE_SOURCE);
    const auto begin =
        text.find(
            "std::optional<BufferCache::PreparedIndexBuffer> BufferCache::PrepareIndexBuffer(");
    ASSERT_NE(begin, std::string::npos);
    const auto end = text.find("bool BufferCache::FinalizeIndexBuffer(", begin);
    ASSERT_NE(end, std::string::npos);
    const auto body = text.substr(begin, end - begin);
    EXPECT_NE(body.find("AmdGpu::TryGetIndexPreparationBytes(regs, index_size, indirect)"),
              std::string::npos);
    EXPECT_NE(body.find("const auto index_buffer_size"), std::string::npos);
    EXPECT_NE(body.find("return std::nullopt"), std::string::npos);
}

TEST(IndexDrawCapacityWiring, PreparationUsesCheckedIndexAddressBeforeCacheLookup) {
    const auto text = Source(SHADPS4_INDEX_DRAW_BUFFER_CACHE_SOURCE);
    const auto begin = text.find(
        "std::optional<BufferCache::PreparedIndexBuffer> BufferCache::PrepareIndexBuffer(");
    const auto end = text.find("bool BufferCache::FinalizeIndexBuffer(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = text.substr(begin, end - begin);
    const auto checked = body.find("AmdGpu::TryGetIndexAddress(");
    const auto prepare = body.find("PrepareBufferBinding(*index_address", checked);
    ASSERT_NE(checked, std::string::npos);
    ASSERT_NE(prepare, std::string::npos);
    EXPECT_LT(checked, prepare);
}

TEST(IndexDrawCapacityWiring, RendererSelectsDirectAndIndirectPreparationCounts) {
    const auto text = Source(SHADPS4_INDEX_DRAW_RASTERIZER_SOURCE);
    const auto direct_begin = text.find("void Rasterizer::Draw(bool");
    const auto indirect_begin = text.find("void Rasterizer::DrawIndirect(");
    const auto dispatch_begin = text.find("void Rasterizer::DispatchDirect(");
    ASSERT_NE(direct_begin, std::string::npos);
    ASSERT_NE(indirect_begin, std::string::npos);
    ASSERT_NE(dispatch_begin, std::string::npos);

    const auto direct = text.substr(direct_begin, indirect_begin - direct_begin);
    const auto indirect = text.substr(indirect_begin, dispatch_begin - indirect_begin);
    EXPECT_NE(direct.find("PrepareIndexBuffer(index_offset, buffer_barriers, false)"),
              std::string::npos);
    EXPECT_NE(indirect.find("PrepareIndexBuffer(0, buffer_barriers, true)"),
              std::string::npos);
    EXPECT_NE(direct.find("index_buffer_prepare_failed = !prepared"), std::string::npos);
    EXPECT_NE(indirect.find("index_buffer_prepare_failed = !prepared"), std::string::npos);
    EXPECT_NE(direct.find("if (index_buffer_prepare_failed)"), std::string::npos);
    EXPECT_NE(indirect.find("if (index_buffer_prepare_failed)"), std::string::npos);
}

TEST(IndexDrawCapacityWiring, IndirectCommandsUseCheckedGuestRangesBeforePreparation) {
    const auto text = Source(SHADPS4_INDEX_DRAW_RASTERIZER_SOURCE);
    const auto draw_begin = text.find("void Rasterizer::DrawIndirect(");
    const auto dispatch_begin = text.find("void Rasterizer::DispatchIndirect(");
    const auto flush_begin = text.find("u64 Rasterizer::Flush(", dispatch_begin);
    ASSERT_NE(draw_begin, std::string::npos);
    ASSERT_NE(dispatch_begin, std::string::npos);
    ASSERT_NE(flush_begin, std::string::npos);
    const auto draw = text.substr(draw_begin, dispatch_begin - draw_begin);
    const auto dispatch = text.substr(dispatch_begin, flush_begin - dispatch_begin);
    EXPECT_NE(draw.find("AmdGpu::TryGetIndirectBufferRange("), std::string::npos);
    EXPECT_NE(dispatch.find("AmdGpu::TryGetIndirectBufferRange("), std::string::npos);
}

} // namespace
