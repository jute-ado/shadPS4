// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <concepts>
#include <limits>

#include <gtest/gtest.h>

#include "video_core/amdgpu/resource.h"
#include "video_core/texture_cache/tile.h"
#include "video_core/texture_cache/types.h"

namespace {

using VideoCore::ApplyCompressedMipCompatibility;
using VideoCore::BuildMicroTiledMipLayout;
using VideoCore::MicroTiledMipLayoutParams;
using VideoCore::SubresourceExtent;
using VideoCore::SubresourceRange;
using VideoCore::TilingWorkgroupCount;

static_assert(std::equality_comparable<SubresourceExtent>);
static_assert(!std::totally_ordered<SubresourceExtent>);

TEST(SubresourceExtent, ContainsEqualExtent) {
    constexpr SubresourceExtent available{.levels = 3, .layers = 6};

    EXPECT_TRUE(available.CanContain(SubresourceExtent{.levels = 3, .layers = 6}));
}

TEST(SubresourceExtent, ContainsSmallerExtentInBothDimensions) {
    constexpr SubresourceExtent available{.levels = 3, .layers = 6};

    EXPECT_TRUE(available.CanContain(SubresourceExtent{.levels = 2, .layers = 5}));
}

TEST(SubresourceExtent, RejectsMoreMipLevelsDespiteFewerLayers) {
    constexpr SubresourceExtent available{.levels = 1, .layers = 6};

    EXPECT_FALSE(available.CanContain(SubresourceExtent{.levels = 3, .layers = 1}));
}

TEST(SubresourceExtent, RejectsMoreLayersDespiteFewerMipLevels) {
    constexpr SubresourceExtent available{.levels = 3, .layers = 1};

    EXPECT_FALSE(available.CanContain(SubresourceExtent{.levels = 1, .layers = 6}));
}

TEST(SubresourceExtent, ExpandsEachDimensionIndependently) {
    constexpr SubresourceExtent available{.levels = 3, .layers = 1};
    constexpr SubresourceExtent requested{.levels = 1, .layers = 6};

    EXPECT_EQ(available.ExpandedToFit(requested), (SubresourceExtent{.levels = 3, .layers = 6}));
}

TEST(SubresourceExtent, ExpansionIsIndependentOfOperandOrder) {
    constexpr SubresourceExtent first{.levels = 3, .layers = 1};
    constexpr SubresourceExtent second{.levels = 1, .layers = 6};

    EXPECT_EQ(first.ExpandedToFit(second), second.ExpandedToFit(first));
}

TEST(SubresourceExtent, KeepsAvailableMipLevel) {
    constexpr SubresourceExtent extent{.levels = 4, .layers = 1};

    EXPECT_EQ(extent.ClampLevel(2), 2u);
}

TEST(SubresourceExtent, ClampsMipLevelToLastAvailableLevel) {
    constexpr SubresourceExtent extent{.levels = 4, .layers = 1};

    EXPECT_EQ(extent.ClampLevel(9), 3u);
}

TEST(SubresourceExtent, ClampsSingleMipImageToLevelZero) {
    constexpr SubresourceExtent extent{.levels = 1, .layers = 1};

    EXPECT_EQ(extent.ClampLevel(9), 0u);
}

TEST(SubresourceRange, FitsAtExactMipAndLayerBoundary) {
    constexpr SubresourceRange range{
        .base = {.level = 1, .layer = 2},
        .extent = {.levels = 2, .layers = 3},
    };

    EXPECT_TRUE(range.FitsWithin(SubresourceExtent{.levels = 3, .layers = 5}));
}

TEST(SubresourceRange, RejectsBaseOutsideAvailableMipLevels) {
    constexpr SubresourceRange range{
        .base = {.level = 3, .layer = 0},
        .extent = {.levels = 1, .layers = 1},
    };

    EXPECT_FALSE(range.FitsWithin(SubresourceExtent{.levels = 3, .layers = 1}));
}

TEST(SubresourceRange, RejectsExtentCrossingAvailableLayers) {
    constexpr SubresourceRange range{
        .base = {.level = 0, .layer = 4},
        .extent = {.levels = 1, .layers = 2},
    };

    EXPECT_FALSE(range.FitsWithin(SubresourceExtent{.levels = 1, .layers = 5}));
}

TEST(SubresourceRange, RejectsOverflowingRange) {
    constexpr SubresourceRange range{
        .base = {.level = std::numeric_limits<u32>::max(), .layer = 0},
        .extent = {.levels = 2, .layers = 1},
    };

    EXPECT_FALSE(range.FitsWithin(
        SubresourceExtent{.levels = std::numeric_limits<u32>::max(), .layers = 1}));
}

TEST(SubresourceRange, RejectsEmptyExtent) {
    constexpr SubresourceRange range{
        .base = {.level = 0, .layer = 0},
        .extent = {.levels = 0, .layers = 1},
    };

    EXPECT_FALSE(range.FitsWithin(SubresourceExtent{.levels = 1, .layers = 1}));
}

TEST(CompressedMipCompatibility, LimitsSampledCompressedViewsToBaseMipWhenEnabled) {
    constexpr SubresourceRange range{
        .base = {.level = 2, .layer = 1},
        .extent = {.levels = 5, .layers = 3},
    };

    EXPECT_EQ(ApplyCompressedMipCompatibility(range, true, false, true),
              (SubresourceRange{
                  .base = {.level = 2, .layer = 1},
                  .extent = {.levels = 1, .layers = 3},
              }));
}

TEST(CompressedMipCompatibility, PreservesUncompressedViews) {
    constexpr SubresourceRange range{
        .base = {.level = 1, .layer = 0},
        .extent = {.levels = 4, .layers = 1},
    };

    EXPECT_EQ(ApplyCompressedMipCompatibility(range, false, false, true), range);
}

TEST(CompressedMipCompatibility, PreservesStorageViews) {
    constexpr SubresourceRange range{
        .base = {.level = 1, .layer = 0},
        .extent = {.levels = 4, .layers = 1},
    };

    EXPECT_EQ(ApplyCompressedMipCompatibility(range, true, true, true), range);
}

TEST(CompressedMipCompatibility, IsInactiveByDefault) {
    constexpr SubresourceRange range{
        .base = {.level = 1, .layer = 0},
        .extent = {.levels = 4, .layers = 1},
    };

    EXPECT_EQ(ApplyCompressedMipCompatibility(range, true, false, false), range);
}

TEST(TilingDispatch, CoversACompressedBlockTailSmallerThanOneWorkgroup) {
    EXPECT_EQ(TilingWorkgroupCount(256, 64), 1u);
}

TEST(TilingDispatch, CoversACompressedBlockTailAfterFullWorkgroups) {
    EXPECT_EQ(TilingWorkgroupCount(1280, 128), 2u);
}

TEST(TilingDispatch, DoesNotDispatchForAnEmptySurface) {
    EXPECT_EQ(TilingWorkgroupCount(0, 64), 0u);
}

TEST(MicroTiledMipLayout, MatchesEightByteCompressedBlockReference) {
    constexpr auto layout = BuildMicroTiledMipLayout(MicroTiledMipLayoutParams{
        .pitch = 1024,
        .height = 1024,
        .depth = 1,
        .levels = 9,
        .layers = 1,
        .bits_per_block = 64,
        .block_compressed = true,
        .pow2_padding = true,
    });
    constexpr std::array<size_t, 9> expected_offsets{
        0x00000, 0x80000, 0xa0000, 0xa8000, 0xaa000, 0xaa800, 0xaaa00, 0xaac00, 0xaae00,
    };
    constexpr std::array<size_t, 9> expected_sizes{
        0x80000, 0x20000, 0x08000, 0x02000, 0x00800, 0x00200, 0x00200, 0x00200, 0x00200,
    };

    for (u32 mip = 0; mip < layout.level_count; ++mip) {
        EXPECT_EQ(layout.levels[mip].offset, expected_offsets[mip]);
        EXPECT_EQ(layout.levels[mip].size, expected_sizes[mip]);
    }
    EXPECT_EQ(layout.total_size, 0xab000u);
}

TEST(MicroTiledMipLayout, MatchesSixteenByteCompressedBlockReference) {
    constexpr auto layout = BuildMicroTiledMipLayout(MicroTiledMipLayoutParams{
        .pitch = 1024,
        .height = 1024,
        .depth = 1,
        .levels = 9,
        .layers = 1,
        .bits_per_block = 128,
        .block_compressed = true,
        .pow2_padding = true,
    });
    constexpr std::array<size_t, 9> expected_offsets{
        0x000000, 0x100000, 0x140000, 0x150000, 0x154000, 0x155000, 0x155400, 0x155800, 0x155c00,
    };
    constexpr std::array<size_t, 9> expected_sizes{
        0x100000, 0x040000, 0x010000, 0x004000, 0x001000, 0x000400, 0x000400, 0x000400, 0x000400,
    };

    for (u32 mip = 0; mip < layout.level_count; ++mip) {
        EXPECT_EQ(layout.levels[mip].offset, expected_offsets[mip]);
        EXPECT_EQ(layout.levels[mip].size, expected_sizes[mip]);
    }
    EXPECT_EQ(layout.total_size, 0x156000u);
}

TEST(MicroTiledMipLayout, PacksUncompressedArrayLayers) {
    constexpr auto layout = BuildMicroTiledMipLayout(MicroTiledMipLayoutParams{
        .pitch = 13,
        .height = 9,
        .depth = 1,
        .levels = 3,
        .layers = 2,
        .bits_per_block = 32,
    });

    EXPECT_EQ(layout.levels[0].pitch, 16u);
    EXPECT_EQ(layout.levels[0].height, 16u);
    EXPECT_EQ(layout.levels[0].size, 2048u);
    EXPECT_EQ(layout.levels[1].offset, 2048u);
    EXPECT_EQ(layout.levels[1].size, 512u);
    EXPECT_EQ(layout.levels[2].offset, 2560u);
    EXPECT_EQ(layout.levels[2].size, 512u);
    EXPECT_EQ(layout.total_size, 3072u);
}

TEST(MicroTiledMipLayout, AlignsThickMipDepth) {
    constexpr auto layout = BuildMicroTiledMipLayout(MicroTiledMipLayoutParams{
        .pitch = 9,
        .height = 9,
        .depth = 5,
        .levels = 3,
        .layers = 2,
        .bits_per_block = 32,
        .thickness = 4,
    });

    EXPECT_EQ(layout.levels[0].size, 16384u);
    EXPECT_EQ(layout.levels[1].offset, 16384u);
    EXPECT_EQ(layout.levels[1].size, 2048u);
    EXPECT_EQ(layout.levels[2].offset, 18432u);
    EXPECT_EQ(layout.levels[2].size, 2048u);
    EXPECT_EQ(layout.total_size, 20480u);
}

TEST(SamplerLodRange, DisablesMipSelectionWhenMipFilteringIsDisabled) {
    AmdGpu::Sampler sampler{};
    sampler.raw0 = u64{256} << 32 | u64{4095} << 44;
    sampler.raw1 = static_cast<u64>(AmdGpu::MipFilter::None) << 26;

    EXPECT_EQ(sampler.EffectiveMinLod(), 0.0f);
    EXPECT_EQ(sampler.EffectiveMaxLod(), 0.0f);
}

TEST(SamplerLodRange, PreservesLodRangeForPointMipFiltering) {
    AmdGpu::Sampler sampler{};
    sampler.raw0 = u64{256} << 32 | u64{768} << 44;
    sampler.raw1 = static_cast<u64>(AmdGpu::MipFilter::Point) << 26;

    EXPECT_EQ(sampler.EffectiveMinLod(), 1.0f);
    EXPECT_EQ(sampler.EffectiveMaxLod(), 3.0f);
}

TEST(SamplerLodRange, PreservesLodRangeForLinearMipFiltering) {
    AmdGpu::Sampler sampler{};
    sampler.raw0 = u64{512} << 32 | u64{1024} << 44;
    sampler.raw1 = static_cast<u64>(AmdGpu::MipFilter::Linear) << 26;

    EXPECT_EQ(sampler.EffectiveMinLod(), 2.0f);
    EXPECT_EQ(sampler.EffectiveMaxLod(), 4.0f);
}

} // namespace
