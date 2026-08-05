// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <span>

#include "common/types.h"

namespace VideoCore {

struct StreamBufferWatch {
    u64 tick{};
    u64 upper_bound{};
};

inline StreamBufferWatch* FindLastCommittedStreamBufferWatch(std::span<StreamBufferWatch> watches,
                                                             std::size_t next_watch_index,
                                                             u64 tick) {
    if (next_watch_index == 0 || watches[next_watch_index].tick != tick) {
        return nullptr;
    }
    return &watches[next_watch_index];
}

} // namespace VideoCore
