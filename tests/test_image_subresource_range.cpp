// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/texture_cache/subresource_range_diagnostic.h"

namespace VideoCore {
namespace {

TEST(ImageSubresourceRange, AcceptsContainedRange) {
    EXPECT_TRUE(IsSubresourceRangeInBounds({.levels = 4, .layers = 6},
                                           {.base_level = 1,
                                            .base_layer = 2,
                                            .levels = 3,
                                            .layers = 4}));
}

TEST(ImageSubresourceRange, RejectsLevelOverflow) {
    EXPECT_FALSE(IsSubresourceRangeInBounds({.levels = 4, .layers = 6},
                                            {.base_level = 3,
                                             .base_layer = 0,
                                             .levels = 2,
                                             .layers = 1}));
}

TEST(ImageSubresourceRange, RejectsLayerOverflow) {
    EXPECT_FALSE(IsSubresourceRangeInBounds({.levels = 4, .layers = 6},
                                            {.base_level = 0,
                                             .base_layer = 5,
                                             .levels = 1,
                                             .layers = 2}));
}

TEST(ImageSubresourceRange, RejectsEmptyRange) {
    EXPECT_FALSE(IsSubresourceRangeInBounds({.levels = 4, .layers = 6},
                                            {.base_level = 0,
                                             .base_layer = 0,
                                             .levels = 0,
                                             .layers = 1}));
}

} // namespace
} // namespace VideoCore
