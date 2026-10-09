// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <optional>

namespace VideoCore {

inline std::size_t PendingStreamBufferWatchesBeforeRotation(
    std::optional<std::size_t> previous_watch_count, std::size_t previous_wait_cursor) {
    // The existing two-generation policy rotates unconditionally, discarding any unreached tail.
    return 0;
}

} // namespace VideoCore
