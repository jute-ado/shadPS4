// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/texture_cache/image_download_policy.h"

namespace VideoCore {
namespace {

TEST(ImageDownloadPolicy, DisabledReadbackNeverQueuesImages) {
    EXPECT_FALSE(ShouldQueueImageDownload(false, false, 1920));
    EXPECT_FALSE(ShouldQueueImageDownload(false, true, 8));
}

TEST(ImageDownloadPolicy, LinearImagesRemainEligibleAtAnyWidth) {
    EXPECT_TRUE(ShouldQueueImageDownload(true, false, 1));
    EXPECT_TRUE(ShouldQueueImageDownload(true, false, 1920));
}

TEST(ImageDownloadPolicy, SmallTiledImagesIncludeTheEightPixelBoundary) {
    EXPECT_TRUE(ShouldQueueImageDownload(true, true, 1));
    EXPECT_TRUE(ShouldQueueImageDownload(true, true, 8));
    EXPECT_FALSE(ShouldQueueImageDownload(true, true, 9));
}

} // namespace
} // namespace VideoCore
