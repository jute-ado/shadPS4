// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

enum class UploadSnapshotClassification {
    Stable,
    CopiedOldState,
    CopiedNewState,
    CopyMismatch,
};

enum class UploadSnapshotPath {
    DirectStream,
    CachedStaging,
};

constexpr UploadSnapshotClassification ClassifyUploadSnapshot(u64 before, u64 copied,
                                                               u64 after) noexcept {
    if (before == copied && copied == after) {
        return UploadSnapshotClassification::Stable;
    }
    if (before == copied) {
        return UploadSnapshotClassification::CopiedOldState;
    }
    if (copied == after) {
        return UploadSnapshotClassification::CopiedNewState;
    }
    return UploadSnapshotClassification::CopyMismatch;
}

void CopyGuestMemoryWithUploadProvenance(VAddr source, u8* destination, u64 size,
                                         UploadSnapshotPath path, u64 scheduler_tick);

} // namespace VideoCore
