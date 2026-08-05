// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace VideoCore {

[[nodiscard]] constexpr bool KeepUploadSnapshotLockedDuringCopy(bool is_gpu_written) noexcept {
    return is_gpu_written;
}

} // namespace VideoCore
