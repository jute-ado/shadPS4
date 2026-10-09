// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <utility>

#include <gtest/gtest.h>

#include "core/libraries/kernel/process.h"
#include "shader_recompiler/resource.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/texture_cache/image_alias_policy.h"
#include "video_core/texture_cache/image_write_history.h"

namespace Libraries::Kernel {
// Environment only: the real ImageInfo constructors/layout and format code are linked.
s32 PS4_SYSV_ABI sceKernelIsNeoMode() {
    return 0;
}
} // namespace Libraries::Kernel

namespace {
using VideoCore::ImageAliasDisposition;
using VideoCore::ImageInfo;
using VideoCore::ImageWriteHistory;
using VideoCore::PlanSameBaseImageAlias;

ImageInfo MakeBcImage(u32 pitch, u32 levels = 12, u32 width = 2048, u32 height = 2048) {
    ImageInfo info{};
    info.type = AmdGpu::ImageType::Color2D;
    info.pixel_format = vk::Format::eBc1RgbaUnormBlock;
    info.props.is_block = true;
    info.props.is_tiled = true;
    info.props.is_pow2 = true;
    info.tile_mode = AmdGpu::TileMode::Thin1DThin;
    info.array_mode = AmdGpu::GetArrayMode(info.tile_mode);
    info.size = {width, height, 1};
    info.pitch = pitch;
    info.num_bits = 64;
    info.resources.levels = levels;
    info.guest_address = 0x100000;
    info.UpdateSize();
    return info;
}

ImageAliasDisposition Plan(const ImageInfo& requested, const ImageInfo& cached,
                           bool old_enough = false, bool gpu_modified = false,
                           ImageWriteHistory history = {}) {
    return PlanSameBaseImageAlias(requested, cached, history, gpu_modified, old_enough);
}

} // namespace

TEST(ImageWriteHistory, NewImageHasNoImageOnlyWrites) {
    const ImageWriteHistory history;
    EXPECT_FALSE(history.MayHaveImageOnlyContents());
}

TEST(ImageWriteHistory, RecordingImageOnlyWriteIsSticky) {
    ImageWriteHistory history;
    history.RecordImageOnlyWrite();
    EXPECT_TRUE(history.MayHaveImageOnlyContents());
    history.RecordImageOnlyWrite();
    EXPECT_TRUE(history.MayHaveImageOnlyContents());
}

TEST(ImageWriteHistory, CopyAndMovePreserveRecordedAuthority) {
    ImageWriteHistory original;
    original.RecordImageOnlyWrite();
    const auto copy = original;
    auto moved = std::move(original);
    EXPECT_TRUE(copy.MayHaveImageOnlyContents());
    EXPECT_TRUE(moved.MayHaveImageOnlyContents());
}

TEST(ImageAliasLayout, ActualBcLayoutExplainsPitchBoundaryAndMipOffsets) {
    const auto old_image = MakeBcImage(2048);
    const auto new_image = MakeBcImage(2049);
    EXPECT_EQ(old_image.guest_size, 0x2ab400U);
    EXPECT_EQ(new_image.guest_size, 0x4ab400U);
    EXPECT_EQ(old_image.mips_layout[0].size, 0x200000U);
    EXPECT_EQ(new_image.mips_layout[0].size, 0x400000U);
    EXPECT_EQ(old_image.mips_layout[0].pitch, 2048U);
    EXPECT_EQ(new_image.mips_layout[0].pitch, 4096U);
    for (u32 mip = 1; mip < 12; ++mip) {
        EXPECT_EQ(old_image.mips_layout[mip].size, new_image.mips_layout[mip].size);
        EXPECT_EQ(old_image.mips_layout[mip].offset + 0x200000, new_image.mips_layout[mip].offset);
    }
}

TEST(ImageAliasLayout, ActualDescriptorConstructorKeepsEncodedPitchPlusOne) {
    auto descriptor = AmdGpu::Image::Null(false);
    descriptor.base_address = 0x100000 >> 8;
    descriptor.data_format = static_cast<u64>(AmdGpu::DataFormat::FormatBc1);
    descriptor.width = 2047;
    descriptor.height = 2047;
    descriptor.pitch = 2047;
    descriptor.last_level = 11;
    descriptor.pow2pad = 1;
    const Shader::ImageResource resource{};
    const ImageInfo first{descriptor, resource};
    descriptor.pitch = 2048;
    const ImageInfo second{descriptor, resource};
    EXPECT_EQ(first.pitch, 2048U);
    EXPECT_EQ(second.pitch, 2049U);
    EXPECT_EQ(first.guest_size, 0x2ab400U);
    EXPECT_EQ(second.guest_size, 0x4ab400U);
    EXPECT_EQ(Plan(first, second), ImageAliasDisposition::Separate);
}

TEST(ImageAliasLayout, LargerPitchCleanRequestIsIndependent) {
    EXPECT_EQ(Plan(MakeBcImage(2049), MakeBcImage(2048)), ImageAliasDisposition::Separate);
}

TEST(ImageAliasLayout, SmallerPitchCleanRequestCannotReuseLargerLayout) {
    EXPECT_EQ(Plan(MakeBcImage(2048), MakeBcImage(2049)), ImageAliasDisposition::Separate);
}

TEST(ImageAliasLayout, OnlyOldCleanIncompatibleImageMayRetire) {
    EXPECT_EQ(Plan(MakeBcImage(2049), MakeBcImage(2048), true), ImageAliasDisposition::RetireClean);
    EXPECT_EQ(Plan(MakeBcImage(2048), MakeBcImage(2049), true), ImageAliasDisposition::RetireClean);
}

TEST(ImageAliasLayout, GpuModifiedIncompatibleImageIsUnsupportedInBothOrders) {
    EXPECT_EQ(Plan(MakeBcImage(2049), MakeBcImage(2048), true, true),
              ImageAliasDisposition::Unsupported);
    EXPECT_EQ(Plan(MakeBcImage(2048), MakeBcImage(2049), true, true),
              ImageAliasDisposition::Unsupported);
}

TEST(ImageAliasLayout, CopiedImageAuthorityPreventsCleanReplacementEvenWithoutGpuFlag) {
    ImageWriteHistory history;
    history.RecordImageOnlyWrite();
    EXPECT_EQ(Plan(MakeBcImage(2049), MakeBcImage(2048), true, false, history),
              ImageAliasDisposition::Unsupported);
    EXPECT_EQ(Plan(MakeBcImage(2048), MakeBcImage(2049), true, false, history),
              ImageAliasDisposition::Unsupported);
}

TEST(ImageAliasLayout, IdenticalLayoutRemainsReusableWithGpuAuthority) {
    ImageWriteHistory history;
    history.RecordImageOnlyWrite();
    const auto info = MakeBcImage(2048);
    EXPECT_EQ(Plan(info, info, true, true, history), ImageAliasDisposition::Compatible);
}

TEST(ImageAliasLayout, DifferentRawPitchesMayHaveEquivalentPaddedLayout) {
    const auto first = MakeBcImage(513, 1, 512, 512);
    const auto second = MakeBcImage(521, 1, 512, 512);
    ASSERT_NE(first.pitch, second.pitch);
    ASSERT_EQ(first.mips_layout[0].pitch, second.mips_layout[0].pitch);
    ASSERT_EQ(first.guest_size, second.guest_size);
    EXPECT_EQ(Plan(first, second), ImageAliasDisposition::Compatible);
    EXPECT_EQ(Plan(second, first), ImageAliasDisposition::Compatible);
}

TEST(ImageAliasLayout, SmallerMipPrefixUsesActualPhysicalOffsets) {
    const auto full = MakeBcImage(2048);
    const auto prefix = MakeBcImage(2048, 3);
    EXPECT_EQ(Plan(prefix, full), ImageAliasDisposition::Compatible);
    EXPECT_EQ(Plan(full, prefix), ImageAliasDisposition::Compatible);
    EXPECT_EQ(Plan(MakeBcImage(2048, 3), MakeBcImage(2049)), ImageAliasDisposition::Separate);
}

TEST(ImageAliasLayout, SmallBcMipDimensionsRemainValid) {
    const auto first = MakeBcImage(3, 2, 3, 3);
    const auto second = MakeBcImage(4, 2, 3, 3);
    ASSERT_EQ(first.mips_layout[0].pitch, 32U);
    ASSERT_EQ(first.mips_layout[1].pitch, 32U);
    ASSERT_EQ(first.guest_size, second.guest_size);
    EXPECT_EQ(Plan(first, second), ImageAliasDisposition::Compatible);
}

TEST(ImageAliasLayout, EqualTotalSizeDoesNotProveMipOffsetCompatibility) {
    const auto first = MakeBcImage(2048);
    auto second = first;
    second.mips_layout[1].offset += 256;
    EXPECT_EQ(Plan(first, second), ImageAliasDisposition::Separate);
    EXPECT_EQ(Plan(second, first), ImageAliasDisposition::Separate);
}

TEST(ImageAliasLayout, TilingSamplesBankAndAlternateInterpretationAreNotViews) {
    const auto first = MakeBcImage(2048);
    auto second = first;
    second.tile_mode = AmdGpu::TileMode::Thin2DThin;
    EXPECT_EQ(Plan(first, second), ImageAliasDisposition::Separate);
    second = first;
    second.num_samples = 2;
    EXPECT_EQ(Plan(first, second), ImageAliasDisposition::Separate);
    second = first;
    second.bank_swizzle = 1;
    EXPECT_EQ(Plan(first, second), ImageAliasDisposition::Separate);
    second = first;
    second.alt_tile = true;
    EXPECT_EQ(Plan(first, second), ImageAliasDisposition::Separate);
}

TEST(ImageAliasLayout, ExplicitDepthAndCompressedUncompressedRulesRemainOutsidePolicy) {
    const auto first = MakeBcImage(2048);
    auto second = first;
    second.props.is_depth = true;
    EXPECT_EQ(Plan(first, second), ImageAliasDisposition::ExistingRules);
    second = first;
    second.props.is_block = false;
    second.pixel_format = vk::Format::eR32G32Uint;
    second.size = {512, 512, 1};
    second.pitch = 512;
    EXPECT_EQ(Plan(first, second), ImageAliasDisposition::ExistingRules);
    EXPECT_EQ(Plan(second, first), ImageAliasDisposition::ExistingRules);
}

TEST(ImageAliasLayout, DifferentAddressesAndVolumeAliasesRetainExistingRules) {
    const auto first = MakeBcImage(2048);
    auto second = first;
    second.guest_address += 4096;
    EXPECT_EQ(Plan(first, second), ImageAliasDisposition::ExistingRules);
    second = first;
    second.type = AmdGpu::ImageType::Color3D;
    EXPECT_EQ(Plan(first, second), ImageAliasDisposition::ExistingRules);
}

TEST(ImageAliasLayout, InvalidMipLayoutsAreUnsupportedInsteadOfNewImagePublication) {
    const auto first = MakeBcImage(2048);
    auto invalid = first;
    invalid.resources.levels = 0;
    EXPECT_EQ(Plan(first, invalid), ImageAliasDisposition::InvalidLayout);
    invalid = first;
    invalid.resources.levels = 17;
    EXPECT_EQ(Plan(first, invalid), ImageAliasDisposition::InvalidLayout);
    invalid = first;
    invalid.mips_layout[0].size = 0;
    EXPECT_EQ(Plan(first, invalid), ImageAliasDisposition::InvalidLayout);
    invalid = first;
    invalid.mips_layout[0].offset = invalid.guest_size;
    EXPECT_EQ(Plan(first, invalid), ImageAliasDisposition::InvalidLayout);
}

TEST(ImageAliasLayout, DebugReceiptsAreDisabledWithoutSpendingAndBoundedWhenEnabled) {
    VideoCore::ImageAliasDebugBudget budget;
    for (u32 attempt = 0; attempt < 100; ++attempt) {
        EXPECT_FALSE(budget.Take(false));
    }
    for (u32 receipt = 0; receipt < 16; ++receipt) {
        EXPECT_TRUE(budget.Take(true));
    }
    for (u32 attempt = 0; attempt < 100; ++attempt) {
        EXPECT_FALSE(budget.Take(true));
    }
}

TEST(ImageAliasLayout, DirtyIncompatibleCandidateDoesNotMaskExactMatchInEitherOrder) {
    const auto requested = MakeBcImage(2048);
    auto incompatible = requested;
    incompatible.bank_swizzle = 1;
    const std::array layouts{incompatible, requested};
    ImageWriteHistory written;
    written.RecordImageOnlyWrite();
    for (const auto order : {std::array{0U, 1U}, std::array{1U, 0U}}) {
        std::optional<u32> selected;
        for (const auto index : order) {
            const auto disposition = Plan(requested, layouts[index], false, index == 0, written);
            if (VideoCore::CanConsiderPerfectImageAlias(disposition)) {
                selected = index;
                break;
            }
        }
        ASSERT_TRUE(selected.has_value());
        EXPECT_EQ(*selected, 1U);
    }
}
