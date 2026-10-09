// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/buffer_cache/upload_snapshot_lock_policy.h"

TEST(UploadSnapshotLockPolicy, KeepsReadOnlyGuestPagesLockedDuringTheCopy) {
    EXPECT_TRUE(VideoCore::KeepUploadSnapshotLockedDuringCopy(false));
}

TEST(UploadSnapshotLockPolicy, KeepsGpuWrittenGuestPagesLockedDuringTheCopy) {
    EXPECT_TRUE(VideoCore::KeepUploadSnapshotLockedDuringCopy(true));
}
