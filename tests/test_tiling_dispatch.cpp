// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/texture_cache/tile.h"

namespace {

using VideoCore::TilingWorkgroupCount;

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
