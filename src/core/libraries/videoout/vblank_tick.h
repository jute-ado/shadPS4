// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::VideoOut {

struct VblankTick {
    u64 count;
    u64 event_data;
};

[[nodiscard]] constexpr VblankTick PrepareVblankTick(u64 prior_count, u64 event_id) {
    const u64 count = prior_count + 1;
    return {
        .count = count,
        .event_data = event_id | (count << 16),
    };
}

} // namespace Libraries::VideoOut
