// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <concepts>

#include <gtest/gtest.h>

#include "video_core/texture_cache/types.h"

namespace {

using VideoCore::SubresourceExtent;

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

} // namespace
