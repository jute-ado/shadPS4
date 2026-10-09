// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <utility>

#include <gtest/gtest.h>

#include "core/libraries/kernel/process.h"
#include "shader_recompiler/resource.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/amdgpu/regs_depth.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/texture_cache/image_alias_policy.h"
#include "video_core/texture_cache/image_write_history.h"
#include "video_core/texture_cache/tile.h"

namespace Libraries::Kernel {
// Environment only: the real ImageInfo constructors/layout and format code are linked.
s32 PS4_SYSV_ABI sceKernelIsNeoMode() {
    return 0;
}
} // namespace Libraries::Kernel

namespace {
using VideoCore::ImageAliasDisposition;
using VideoCore::ImageLayoutCompatibility;
using VideoCore::ImageInfo;
using VideoCore::ImageWriteHistory;
using VideoCore::PlanSameBaseImageAlias;
using VideoCore::ComparePhysicalImageLayout;

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

TEST(ImageAliasLayout, ResourceUnionRecomputesLayerFootprint) {
    const auto requested = MakeBcImage(2048, 1);
    auto cached = requested;
    cached.resources.layers = 6;
    cached.UpdateSize();

    auto recreated = requested;
    recreated.resources =
        VideoCore::MergeSubresourceExtents(requested.resources, cached.resources);
    recreated.UpdateSize();

    EXPECT_EQ(recreated.resources.levels, 1U);
    EXPECT_EQ(recreated.resources.layers, 6U);
    EXPECT_EQ(recreated.mips_layout[0].size, cached.mips_layout[0].size);
    EXPECT_EQ(recreated.guest_size, cached.guest_size);
    EXPECT_TRUE(recreated.HasValidMipChain());
}

TEST(ImageAliasLayout, BlockCompressedAltTileUsesBlockBitsForBankSwizzle) {
    auto descriptor = AmdGpu::Image::Null(false);
    descriptor.base_address = 0x70;
    descriptor.data_format = static_cast<u64>(AmdGpu::DataFormat::FormatBc1);
    descriptor.num_format = static_cast<u64>(AmdGpu::NumberFormat::Unorm);
    descriptor.tiling_index = static_cast<u64>(AmdGpu::TileMode::Thin2DThin);
    descriptor.width = 2047;
    descriptor.height = 2047;
    descriptor.pitch = 2047;
    descriptor.alt_tile_mode = 1;

    const Shader::ImageResource resource{};
    const ImageInfo info{descriptor, resource};

    // Bank-swizzle calculation consumes the bits-per-block value used by the
    // detiler.  BC1 is four bits per texel but 64 bits per 4x4 block; using
    // the former underflows CalculateMacrotileMode and reaches the unreachable
    // unknown-macro-tile path.
    EXPECT_EQ(info.num_bits, 64U);
    EXPECT_EQ(info.bank_swizzle, 7U);
    EXPECT_TRUE(info.HasValidMipChain());
}

TEST(ImageAliasLayout, LinearGeneralDescriptorUsesUnalignedGuestGeometry) {
    auto descriptor = AmdGpu::Image::Null(false);
    descriptor.base_address = 0x100000 >> 8;
    descriptor.tiling_index = static_cast<u64>(AmdGpu::TileMode::DisplayLinearGeneral);
    descriptor.width = 7;
    descriptor.height = 3;
    descriptor.pitch = 11;
    descriptor.last_level = 0;

    const Shader::ImageResource resource{};
    const ImageInfo info{descriptor, resource};

    EXPECT_FALSE(info.IsTiled());
    EXPECT_EQ(info.mips_layout[0].pitch, 12U);
    EXPECT_EQ(info.mips_layout[0].height, 4U);
    EXPECT_EQ(info.mips_layout[0].size, 192U);
    EXPECT_EQ(info.guest_size, 192U);
}

TEST(ImageAliasLayout, OneDimensionalDescriptorUsesVulkanOneDimensionalExtent) {
    // The height field is unused by the PS4 1D image descriptor.  Some Shadow
    // descriptors carry stale data there (the observed value was 2179), but a
    // Vulkan VK_IMAGE_TYPE_1D image must always use height=depth=1.
    for (const auto image_type : {AmdGpu::ImageType::Color1D,
                                  AmdGpu::ImageType::Color1DArray}) {
        auto descriptor = AmdGpu::Image::Null(false);
        descriptor.base_address = 0x100000 >> 8;
        descriptor.type = static_cast<u64>(image_type);
        descriptor.width = 1023;
        descriptor.height = 2178;
        descriptor.depth = image_type == AmdGpu::ImageType::Color1DArray ? 2 : 0;

        const Shader::ImageResource resource{};
        const ImageInfo info{descriptor, resource};

        EXPECT_EQ(info.type, AmdGpu::ImageType::Color1D);
        EXPECT_EQ(info.size.width, 1024U);
        EXPECT_EQ(info.size.height, 1U);
        EXPECT_EQ(info.size.depth, 1U);
        EXPECT_EQ(info.resources.layers,
                  image_type == AmdGpu::ImageType::Color1DArray ? 3U : 1U);
    }
}

TEST(ImageAliasLayout, DescriptorFootprintOverflowFailsClosed) {
    auto descriptor = AmdGpu::Image::Null(false);
    descriptor.base_address = 0x100000 >> 8;
    descriptor.type = static_cast<u64>(AmdGpu::ImageType::Color2DArray);
    descriptor.tiling_index = static_cast<u64>(AmdGpu::TileMode::DisplayLinearGeneral);
    descriptor.width = 16383;
    descriptor.height = 16383;
    descriptor.pitch = 16383;
    descriptor.depth = 8191;
    descriptor.last_level = 0;

    const Shader::ImageResource resource{};
    const ImageInfo info{descriptor, resource};

    // A single 16K x 16K 32-bit slice times 8192 layers is far beyond the u32 guest-size field.
    // The descriptor is otherwise structurally valid, so this specifically exercises checked
    // footprint arithmetic rather than the mip-chain or tile-mode guards.
    EXPECT_FALSE(info.HasValidMipChain());
}

TEST(ImageAliasLayout, MipLayoutCapacityFailsClosed) {
    ImageInfo info{};
    info.type = AmdGpu::ImageType::Color2D;
    info.tile_mode = AmdGpu::TileMode::DisplayLinearGeneral;
    info.array_mode = AmdGpu::ArrayMode::ArrayLinearGeneral;
    info.size = {1u << 16, 1, 1};
    info.pitch = 1u << 16;
    info.num_bits = 32;
    info.resources.levels = static_cast<u32>(info.mips_layout.size()) + 1;
    info.resources.layers = 1;

    // The extent supports this many complete levels, but ImageInfo only has a
    // fixed-size layout table.  Admission must reject the descriptor before
    // UpdateSize can index beyond that table.
    EXPECT_FALSE(info.HasValidMipChain());

    info.UpdateSize();
    EXPECT_FALSE(info.has_valid_guest_size);
    EXPECT_EQ(info.guest_size, 0U);

    info.resources.levels = static_cast<u32>(info.mips_layout.size());
    info.UpdateSize();
    EXPECT_TRUE(info.has_valid_guest_size);
    EXPECT_TRUE(info.HasValidMipChain());
}

TEST(ImageAliasLayout, LinearAlignedFootprintUsesWideIntermediate) {
    const auto [pitch, height, size] = VideoCore::ImageSizeLinearAligned(16384, 16384, 32, 16);
    EXPECT_EQ(pitch, 16384U);
    EXPECT_EQ(height, 16384U);
    EXPECT_EQ(size, size_t{17179869184ULL});
}

TEST(ImageAliasLayout, ColorRegisterSliceFootprintDoesNotWrapBeforeAdmission) {
    AmdGpu::ColorBuffer buffer{};
    buffer.base_address = 0x100;
    buffer.pitch.tile_max = 0;
    buffer.slice.tile_max = (1u << 22) - 1;
    buffer.view.slice_max = 0;
    buffer.info.format = static_cast<u32>(AmdGpu::DataFormat::Format32_32_32_32);
    buffer.attrib.tile_mode_index = AmdGpu::TileMode::Thin2DThin;

    // 16 bytes * 4,194,304 tile rows * 64 bytes = exactly 2^32 bytes.  The
    // register helper must retain that wide result so ImageInfo can reject it
    // as unrepresentable in its u32 guest range.
    EXPECT_EQ(buffer.GetColorSliceSize(), 1ULL << 32);

    AmdGpu::CbDbExtent hint{};
    hint.width = 1;
    hint.height = 1;
    const ImageInfo info{buffer, hint};
    EXPECT_FALSE(info.HasValidMipChain());
    EXPECT_EQ(info.guest_size, 0U);
}

TEST(ImageAliasLayout, DepthRegisterSliceFootprintDoesNotWrapBeforeAdmission) {
    AmdGpu::DepthBuffer buffer{};
    buffer.z_read_base = 0x100;
    buffer.z_info.format = AmdGpu::DepthBuffer::ZFormat::Z32Float;
    buffer.z_info.num_samples = 3;
    buffer.z_info.tile_mode_index = static_cast<u32>(AmdGpu::TileMode::Depth2DThin64);
    buffer.depth_size.pitch_tile_max = 0;
    buffer.depth_size.height_tile_max = 0;
    buffer.depth_slice.tile_max = (1u << 22) - 1;

    // 4 bytes * 4,194,304 tile rows * 64 bytes * 8 samples = 2^33 bytes.
    EXPECT_EQ(buffer.GetDepthSliceSize(), 1ULL << 33);

    AmdGpu::CbDbExtent hint{};
    hint.width = 1;
    hint.height = 1;
    const ImageInfo info{buffer, 1, 0, hint};
    EXPECT_FALSE(info.HasValidMipChain());
    EXPECT_EQ(info.guest_size, 0U);
}

TEST(ImageTiling, RotatedGfx8TileModesHaveKnownLayouts) {
    using AmdGpu::ArrayMode;
    using AmdGpu::PipeConfig;
    using AmdGpu::TileMode;

    struct Expected {
        TileMode tile_mode;
        ArrayMode array_mode;
        u32 sample_split;
    };
    constexpr std::array<Expected, 3> expected{{
        Expected{static_cast<TileMode>(27), ArrayMode::Array1DTiledThin1, 2},
        Expected{static_cast<TileMode>(28), ArrayMode::Array2DTiledThin1, 2},
        Expected{static_cast<TileMode>(29), ArrayMode::ArrayPrtTiledThin1, 8},
    }};

    for (const auto& item : expected) {
        const auto actual = AmdGpu::TryGetArrayMode(item.tile_mode);
        ASSERT_TRUE(actual.has_value()) << "tile mode " << static_cast<u32>(item.tile_mode);
        EXPECT_EQ(*actual, item.array_mode);
        EXPECT_EQ(AmdGpu::GetMicroTileMode(item.tile_mode), AmdGpu::MicroTileMode::Rotated);
        EXPECT_EQ(AmdGpu::GetPipeConfig(item.tile_mode), PipeConfig::P8_32x32_16x16);
        EXPECT_EQ(AmdGpu::GetAltPipeConfig(item.tile_mode), PipeConfig::P16_32x32_8x16);
        EXPECT_EQ(AmdGpu::GetSampleSplit(item.tile_mode), item.sample_split);
        EXPECT_EQ(AmdGpu::GetTileSplitHw(item.tile_mode), 64U);
    }
}

TEST(ImageTiling, RotatedMacroTileModesReuseInBoundsCanonicalExtents) {
    using AmdGpu::TileMode;

    struct Expected {
        TileMode rotated;
        TileMode canonical;
    };
    constexpr std::array expected{
        Expected{TileMode::Rotated2DThin, TileMode::Thin2DThin},
        Expected{TileMode::RotatedThinPrt, TileMode::ThinThinPrt},
    };

    constexpr std::array bpps{8U, 16U, 32U, 64U, 128U};
    constexpr std::array samples{1U, 2U, 4U, 8U};
    for (const auto& item : expected) {
        for (const auto bpp : bpps) {
            for (const auto num_samples : samples) {
                for (const bool alt : {false, true}) {
                    EXPECT_EQ(VideoCore::GetMacroTileExtents(item.rotated, bpp, num_samples, alt),
                              VideoCore::GetMacroTileExtents(item.canonical, bpp, num_samples, alt))
                        << "rotated tile mode " << static_cast<u32>(item.rotated) << " bpp " << bpp
                        << " samples " << num_samples << " alt " << alt;
                }
            }
        }
    }
}

TEST(ImageTiling, ReservedGfx8TileModeRemainsUnknown) {
    EXPECT_FALSE(AmdGpu::TryGetArrayMode(static_cast<AmdGpu::TileMode>(30)));
}

TEST(ImageAliasLayout, ReservedDescriptorTileModeFailsClosed) {
    auto descriptor = AmdGpu::Image::Null(false);
    descriptor.base_address = 0x100000 >> 8;
    descriptor.tiling_index = 30;

    const Shader::ImageResource resource{};
    const ImageInfo info{descriptor, resource};

    EXPECT_FALSE(info.HasValidTileMode());
    EXPECT_FALSE(info.HasValidMipChain());
    EXPECT_EQ(info.pixel_format, vk::Format::eUndefined);
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

TEST(ImageAliasLayout, PhysicalLayoutComparisonRejectsEqualSizeMipOffsetMismatch) {
    const auto first = MakeBcImage(2048);
    auto second = first;
    second.mips_layout[1].offset += 256;

    EXPECT_EQ(ComparePhysicalImageLayout(first, second), ImageLayoutCompatibility::Incompatible);
    EXPECT_EQ(ComparePhysicalImageLayout(second, first), ImageLayoutCompatibility::Incompatible);
}

TEST(ImageAliasLayout, PhysicalLayoutComparisonAcceptsExactMipPrefix) {
    const auto full = MakeBcImage(2048);
    const auto prefix = MakeBcImage(2048, 3);

    EXPECT_EQ(ComparePhysicalImageLayout(prefix, full), ImageLayoutCompatibility::Compatible);
    EXPECT_EQ(ComparePhysicalImageLayout(full, prefix), ImageLayoutCompatibility::Compatible);
}

TEST(ImageAliasLayout, LargerRequestRejectsMismatchInCommonMipPrefix) {
    auto full = MakeBcImage(2048);
    const auto prefix = MakeBcImage(2048, 3);
    full.mips_layout[1].offset += 256;

    EXPECT_EQ(ComparePhysicalImageLayout(full, prefix), ImageLayoutCompatibility::Incompatible);
}

TEST(ImageAliasLayout, PhysicalLayoutComparisonLeavesLegacyNoMipMetadataUnknown) {
    ImageInfo color_buffer{};
    color_buffer.guest_address = 0x100000;
    color_buffer.guest_size = 4096;
    auto other = color_buffer;
    other.guest_size = 8192;

    EXPECT_EQ(ComparePhysicalImageLayout(color_buffer, other), ImageLayoutCompatibility::Unknown);
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

TEST(ImageTiling, RotatedGfx8TileModesHaveKnownArrayLayouts) {
    using AmdGpu::ArrayMode;
    using AmdGpu::TileMode;

    struct Expected {
        TileMode tile_mode;
        ArrayMode array_mode;
    };
    constexpr std::array expected{
        Expected{static_cast<TileMode>(27), ArrayMode::Array1DTiledThin1},
        Expected{static_cast<TileMode>(28), ArrayMode::Array2DTiledThin1},
        Expected{static_cast<TileMode>(29), ArrayMode::ArrayPrtTiledThin1},
    };

    for (const auto& item : expected) {
        const auto actual = AmdGpu::TryGetArrayMode(item.tile_mode);
        ASSERT_TRUE(actual.has_value()) << "tile mode " << static_cast<u32>(item.tile_mode);
        EXPECT_EQ(*actual, item.array_mode)
            << "tile mode " << static_cast<u32>(item.tile_mode);
    }
}
