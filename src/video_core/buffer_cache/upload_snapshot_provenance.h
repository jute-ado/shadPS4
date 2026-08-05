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

constexpr UploadSnapshotClassification ClassifyUploadSnapshot(u64 before, u64 copied,
                                                               u64 after) noexcept {
    static_cast<void>(before);
    static_cast<void>(copied);
    static_cast<void>(after);
    return UploadSnapshotClassification::Stable;
}

} // namespace VideoCore
