// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <concepts>
#include <limits>

#include <gtest/gtest.h>

#include "video_core/texture_cache/tile.h"
#include "video_core/texture_cache/types.h"

namespace {

using VideoCore::ApplyCompressedMipCompatibility;
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

} // namespace
