// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "video_core/texture_cache/image_tiling_range.h"

namespace {

VideoCore::ImageInfo MakeImage() {
    VideoCore::ImageInfo info{};
    info.props.is_tiled = true;
    info.resources = {2, 1};
    info.size = {128, 128, 1};
    info.num_bits = 32;
    info.pitch = 128;
    info.guest_size = 80_KB;
    info.mips_layout[0] = {64_KB, 128, 128, 0};
    info.mips_layout[1] = {16_KB, 64, 64, 64_KB};
    return info;
}

vk::BufferImageCopy MakeCopy(const VideoCore::ImageInfo& info, u32 mip) {
    return {
        .bufferOffset = info.mips_layout[mip].offset,
        .bufferRowLength = info.mips_layout[mip].pitch,
        .bufferImageHeight = info.mips_layout[mip].height,
        .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .mipLevel = mip,
                             .baseArrayLayer = 0,
                             .layerCount = info.resources.layers},
        .imageOffset = {0, 0, 0},
        .imageExtent = {std::max(info.size.width >> mip, 1U), std::max(info.size.height >> mip, 1U),
                        std::max(info.size.depth >> mip, 1U)},
    };
}

TEST(ImageTilingRange, PartialMipPrefixDoesNotBindOrDispatchTheUncopiedSuffix) {
    const auto info = MakeImage();
    const std::array copies{MakeCopy(info, 0)};
    const auto plan = VideoCore::PlanImageTilingRange(info, copies, 0, 64_KB);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->byte_size, 64_KB);
    EXPECT_EQ(plan->element_count, 16384U);
    EXPECT_EQ(plan->dispatch_x, 256U);
    EXPECT_LE(plan->byte_size, 64_KB);
    EXPECT_EQ(plan->num_mips, 1U);
}

TEST(ImageTilingRange, CompleteImagePreservesTheExistingRangeAndWorkSize) {
    const auto info = MakeImage();
    const std::array copies{MakeCopy(info, 0), MakeCopy(info, 1)};
    const auto plan = VideoCore::PlanImageTilingRange(info, copies, 0, 80_KB);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->byte_size, 80_KB);
    EXPECT_EQ(plan->element_count, 20480U);
    EXPECT_EQ(plan->dispatch_x, 320U);
    EXPECT_EQ(plan->num_mips, 2U);
}

TEST(ImageTilingRange, NonzeroOutputOffsetIsAppliedExactlyOnce) {
    const auto info = MakeImage();
    const std::array copies{MakeCopy(info, 0)};
    const auto plan = VideoCore::PlanImageTilingRange(info, copies, 16_KB, 80_KB);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->byte_size, 64_KB);
    EXPECT_EQ(plan->mips[0].offset, 0U);
    EXPECT_EQ(copies[0].bufferOffset, 0U);
    EXPECT_FALSE(VideoCore::PlanImageTilingRange(info, copies, 16_KB + 1, 80_KB));
}

TEST(ImageTilingRange, RejectsOutputOffsetPastEndWithoutWrapping) {
    const auto info = MakeImage();
    const std::array copies{MakeCopy(info, 0)};
    EXPECT_FALSE(VideoCore::PlanImageTilingRange(info, copies, 80_KB + 1, 80_KB));
    EXPECT_FALSE(
        VideoCore::PlanImageTilingRange(info, copies, std::numeric_limits<u64>::max() - 1, 80_KB));
    EXPECT_FALSE(VideoCore::PlanImageTilingRange(info, copies, 0, 64_KB - 1));
}

TEST(ImageTilingRange, BlockCompressedArrayCountsBlocksAndIncludesEveryLayer) {
    auto info = MakeImage();
    info.props.is_block = true;
    info.resources.layers = 3;
    info.size = {64, 32, 1};
    info.pitch = 64;
    info.num_bits = 64;
    info.guest_size = 3840;
    info.mips_layout[0] = {3072, 64, 32, 0};
    info.mips_layout[1] = {768, 32, 16, 3072};
    const std::array copies{MakeCopy(info, 0)};
    const auto plan = VideoCore::PlanImageTilingRange(info, copies, 256, 3328);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->byte_size, 3072U);
    EXPECT_EQ(plan->element_count, 384U);
    EXPECT_EQ(plan->dispatch_x, 6U);
    EXPECT_EQ(plan->num_slices, 3U);
    EXPECT_EQ(plan->mips[0].pitch, 16U);
    EXPECT_EQ(plan->mips[0].height, 8U);
    EXPECT_EQ(plan->mips[0].size, 3072U);
}

TEST(ImageTilingRange, VolumePrefixUsesFullFirstMipDepthAndShrinkingSecondMip) {
    auto info = MakeImage();
    info.props.is_volume = true;
    info.size = {64, 32, 4};
    info.pitch = 64;
    info.guest_size = 36_KB;
    info.mips_layout[0] = {32_KB, 64, 32, 0};
    info.mips_layout[1] = {4_KB, 32, 16, 32_KB};
    const std::array first{MakeCopy(info, 0)};
    const auto prefix = VideoCore::PlanImageTilingRange(info, first, 4_KB, 36_KB);
    ASSERT_TRUE(prefix);
    EXPECT_EQ(prefix->byte_size, 32_KB);
    EXPECT_EQ(prefix->num_slices, 4U);
    const std::array full{MakeCopy(info, 0), MakeCopy(info, 1)};
    const auto complete = VideoCore::PlanImageTilingRange(info, full, 4_KB, 40_KB);
    ASSERT_TRUE(complete);
    EXPECT_EQ(complete->byte_size, 36_KB);
    EXPECT_EQ(complete->mips[1].offset, 32_KB);
    EXPECT_EQ(full[1].imageExtent.depth, 2U);
}

TEST(ImageTilingRange, AcceptsSmallPrefixAndRoundsUpItsLaunch) {
    auto info = MakeImage();
    info.resources.levels = 1;
    info.size = {1, 1, 1};
    info.pitch = 1;
    info.guest_size = 260;
    info.mips_layout[0] = {260, 1, 1, 0};
    const std::array copies{MakeCopy(info, 0)};
    const auto plan = VideoCore::PlanImageTilingRange(info, copies, 0, 260);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->element_count, 65U);
    EXPECT_EQ(plan->dispatch_x, 2U);
}

TEST(ImageTilingRange, LinearPrefixUsesSameBoundsWithoutChangingByteAddressing) {
    auto info = MakeImage();
    info.props.is_tiled = false;
    const std::array copies{MakeCopy(info, 0)};
    const auto plan = VideoCore::PlanImageTilingRange(info, copies, 16_KB, 80_KB);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->byte_size, 64_KB);
    EXPECT_EQ(plan->mips[0].pitch, 128U);
}

TEST(ImageTilingRange, SupportedNinetySixBitElementsKeepTheirExistingStride) {
    auto info = MakeImage();
    info.num_bits = 96;
    info.guest_size *= 3;
    info.mips_layout[0].size *= 3;
    info.mips_layout[1].size *= 3;
    info.mips_layout[1].offset *= 3;
    const std::array copies{MakeCopy(info, 0)};
    const auto plan = VideoCore::PlanImageTilingRange(info, copies, 0, 192_KB);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->byte_size, 192_KB);
    EXPECT_EQ(plan->element_count, 16384U);
    EXPECT_EQ(plan->dispatch_x, 256U);
}

TEST(ImageTilingRange, OutputOffsetRemainsWideInsteadOfTruncatingToU32) {
    const auto info = MakeImage();
    const std::array copies{MakeCopy(info, 0)};
    constexpr u64 offset = u64{1} << 32;
    const auto plan = VideoCore::PlanImageTilingRange(info, copies, offset, offset + 64_KB);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->byte_size, 64_KB);
    EXPECT_FALSE(VideoCore::PlanImageTilingRange(info, copies, offset, 64_KB));
}

TEST(ImageTilingRange, UncopiedMipsDoNotSupplyGeometryOrWork) {
    auto info = MakeImage();
    info.mips_layout[1] = {0, 0, 0, 0};
    const std::array copies{MakeCopy(info, 0)};
    const auto plan = VideoCore::PlanImageTilingRange(info, copies, 0, 64_KB);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->num_mips, 1U);
    EXPECT_EQ(plan->mips[1].size, 0U);
    EXPECT_EQ(plan->byte_size, 64_KB);
}

// Numerical counterpart of the shader loop, pinned to its actual bounds checks below.
u32 LookupMip(const VideoCore::ImageTilingPlan& plan, u32 element_bytes, u32 texel) {
    u32 mip = 0;
    while (mip < plan.num_mips) {
        const u32 mip_size = plan.mips[mip].size / element_bytes;
        if (texel < mip_size) {
            return mip;
        }
        texel -= mip_size;
        ++mip;
    }
    return mip;
}

TEST(ImageTilingRange, ShaderGuardRejectsRoundedTailAndEmptyPrefixBeforeGeometry) {
    VideoCore::ImageTilingPlan empty{};
    EXPECT_EQ(LookupMip(empty, 4, 0), empty.num_mips);
    auto info = MakeImage();
    info.resources.levels = 1;
    info.size = {1, 1, 1};
    info.mips_layout[0] = {260, 1, 1, 0};
    info.guest_size = 260;
    const std::array copies{MakeCopy(info, 0)};
    const auto plan = VideoCore::PlanImageTilingRange(info, copies, 0, 260);
    ASSERT_TRUE(plan);
    u32 accepted = 0;
    u32 rejected = 0;
    for (u32 texel = 0; texel < plan->dispatch_x * 64; ++texel) {
        const u32 mip = LookupMip(*plan, 4, texel);
        if (mip >= plan->num_mips) {
            ++rejected;
            continue;
        }
        ASSERT_NE(plan->mips[mip].pitch, 0U);
        ASSERT_NE(plan->mips[mip].height, 0U);
        ++accepted;
    }
    EXPECT_EQ(accepted, 65U);
    EXPECT_EQ(rejected, 63U);
}

TEST(ImageTilingRange, ShaderGuardDoesNotReadOnePastTheSixteenMipUniform) {
    VideoCore::ImageTilingPlan plan{};
    plan.num_mips = 16;
    for (auto& mip : plan.mips) {
        mip = {256, 8, 8, 0};
    }
    EXPECT_EQ(LookupMip(plan, 4, 1023), 15U);
    EXPECT_EQ(LookupMip(plan, 4, 1024), 16U);
    EXPECT_EQ(LookupMip(plan, 4, std::numeric_limits<u32>::max()), 16U);
}

TEST(ImageTilingRange, RejectsEmptyOrExcessiveMipLists) {
    auto info = MakeImage();
    EXPECT_FALSE(VideoCore::PlanImageTilingRange(info, {}, 0, 80_KB));
    std::vector<vk::BufferImageCopy> copies(17, MakeCopy(info, 0));
    EXPECT_FALSE(VideoCore::PlanImageTilingRange(info, copies, 0, 80_KB));
    copies.resize(2);
    copies[1] = MakeCopy(info, 1);
    info.resources.levels = 1;
    EXPECT_FALSE(VideoCore::PlanImageTilingRange(info, copies, 0, 80_KB));
    info.resources.levels = 0;
    copies.resize(1);
    EXPECT_FALSE(VideoCore::PlanImageTilingRange(info, copies, 0, 80_KB));
}

TEST(ImageTilingRange, RejectsNonPrefixOrPartialSubresources) {
    const auto info = MakeImage();
    const auto valid_copy = MakeCopy(info, 0);
    for (u32 mutation = 0; mutation < 8; ++mutation) {
        SCOPED_TRACE(mutation);
        std::array copies{valid_copy};
        switch (mutation) {
        case 0:
            copies[0].imageSubresource.mipLevel = 1;
            break;
        case 1:
            copies[0].imageSubresource.baseArrayLayer = 1;
            break;
        case 2:
            copies[0].imageSubresource.layerCount = 0;
            break;
        case 3:
            copies[0].imageOffset.x = 1;
            break;
        case 4:
            copies[0].imageExtent.width -= 1;
            break;
        case 5:
            copies[0].bufferOffset += 4;
            break;
        case 6:
            copies[0].bufferRowLength += 1;
            break;
        case 7:
            copies[0].bufferImageHeight += 1;
            break;
        }
        EXPECT_FALSE(VideoCore::PlanImageTilingRange(info, copies, 0, 80_KB));
    }
}

TEST(ImageTilingRange, RejectsMissingArrayLayersAndVolumeSlices) {
    auto info = MakeImage();
    info.resources.layers = 2;
    info.guest_size *= 2;
    info.mips_layout[0].size *= 2;
    info.mips_layout[1].size *= 2;
    info.mips_layout[1].offset *= 2;
    std::array copies{MakeCopy(info, 0)};
    copies[0].imageSubresource.layerCount = 1;
    EXPECT_FALSE(VideoCore::PlanImageTilingRange(info, copies, 0, info.guest_size));
    info = MakeImage();
    info.props.is_volume = true;
    info.size.depth = 2;
    copies[0] = MakeCopy(info, 0);
    copies[0].imageExtent.depth = 1;
    EXPECT_FALSE(VideoCore::PlanImageTilingRange(info, copies, 0, info.guest_size));
}

TEST(ImageTilingRange, RejectsInvalidMipStorageBeforeBuildingAPlan) {
    for (u32 mutation = 0; mutation < 7; ++mutation) {
        SCOPED_TRACE(mutation);
        auto info = MakeImage();
        switch (mutation) {
        case 0:
            info.mips_layout[0].size = 0;
            break;
        case 1:
            info.mips_layout[0].pitch = 0;
            break;
        case 2:
            info.mips_layout[0].height = 0;
            break;
        case 3:
            info.mips_layout[0].size = 16;
            break;
        case 4:
            info.mips_layout[0].size += 1;
            break;
        case 5:
            info.mips_layout[1].offset += 4;
            break;
        case 6:
            info.guest_size -= 1;
            break;
        }
        const std::array copies{MakeCopy(info, 0), MakeCopy(info, 1)};
        EXPECT_FALSE(
            VideoCore::PlanImageTilingRange(info, copies, 0, std::numeric_limits<u64>::max()));
    }
}

TEST(ImageTilingRange, RejectsInvalidElementAndImageGeometry) {
    for (u32 mutation = 0; mutation < 7; ++mutation) {
        SCOPED_TRACE(mutation);
        auto info = MakeImage();
        switch (mutation) {
        case 0:
            info.num_bits = 0;
            break;
        case 1:
            info.num_bits = 24;
            break;
        case 2:
            info.size.width = 0;
            break;
        case 3:
            info.size.height = 0;
            break;
        case 4:
            info.size.depth = 0;
            break;
        case 5:
            info.resources.layers = 0;
            break;
        case 6:
            info.num_samples = 0;
            break;
        }
        const std::array copies{MakeCopy(info, 0)};
        EXPECT_FALSE(VideoCore::PlanImageTilingRange(info, copies, 0, 80_KB));
    }
}

std::string ReadText(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

TEST(ImageTilingRangeWiring, ProductionCommandsConsumeTheValidatedPrefix) {
    const auto source = ReadText(SHADPS4_TILE_MANAGER_SOURCE_PATH);
    const auto begin = source.find("bool TileManager::TileImage(");
    ASSERT_NE(begin, std::string::npos);
    const auto body = source.substr(begin);
    EXPECT_NE(body.find("PlanImageTilingRange(info, buffer_copies, out_offset, out_size)"),
              std::string::npos);
    EXPECT_LT(body.find("if (!plan)"), body.find("GetScratchBuffer("));
    EXPECT_NE(body.find("GetScratchBuffer(plan->byte_size)"), std::string::npos);
    EXPECT_EQ(body.find(".range = info.guest_size"), std::string::npos);
    EXPECT_NE(body.find("params.mips = plan->mips"), std::string::npos);
    EXPECT_NE(body.find("cmdbuf.dispatch(plan->dispatch_x, 1, 1)"), std::string::npos);
    const auto caller = ReadText(SHADPS4_BUFFER_CACHE_SOURCE_PATH);
    const auto call = caller.find("return tile_manager.TileImage(");
    ASSERT_NE(call, std::string::npos);
    EXPECT_NE(caller.substr(call, 180).find("buffer.SizeBytes()"), std::string::npos);
}

TEST(ImageTilingRangeWiring, ShaderChecksMipLimitBeforeLookupAndBeforeDivision) {
    const auto source = ReadText(SHADPS4_TILING_SHADER_SOURCE_PATH);
    const auto begin = source.find("uint GetMipLevel(");
    const auto end = source.find("void main()", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto lookup = source.substr(begin, end - begin);
    EXPECT_LT(lookup.find("while (mip < info.num_mips)"), lookup.find("info.mips[mip].size"));
    const auto main = source.substr(end);
    EXPECT_LT(main.find("if (mip >= info.num_mips)"), main.find("info.mips[mip].pitch"));
    EXPECT_LT(main.find("return;"), main.find("texel % pitch"));
}

} // namespace
