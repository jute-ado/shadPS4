// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/buffer_cache/upload_snapshot_provenance.h"

namespace {

using VideoCore::ClassifyUploadSnapshot;
using VideoCore::UploadSnapshotClassification;

TEST(UploadSnapshotProvenance, ClassifiesStableCopy) {
    EXPECT_EQ(ClassifyUploadSnapshot(1, 1, 1), UploadSnapshotClassification::Stable);
}

TEST(UploadSnapshotProvenance, ClassifiesCopyOfOldSourceState) {
    EXPECT_EQ(ClassifyUploadSnapshot(1, 1, 2), UploadSnapshotClassification::CopiedOldState);
}

TEST(UploadSnapshotProvenance, ClassifiesCopyOfNewSourceState) {
    EXPECT_EQ(ClassifyUploadSnapshot(1, 2, 2), UploadSnapshotClassification::CopiedNewState);
}

TEST(UploadSnapshotProvenance, ClassifiesCopyMismatch) {
    EXPECT_EQ(ClassifyUploadSnapshot(1, 3, 2), UploadSnapshotClassification::CopyMismatch);
}

} // namespace
