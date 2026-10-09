// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <span>

#include "common/types.h"

namespace VideoCore {

// Owns the original user-data prefix and the first successful SRT-walker output. This is only
// diagnostic evidence: it must never trigger another guest-memory read after a failed walk.
struct SrtFailureFlatSnapshot {
    static constexpr u32 PrefixDwords = 22;

    std::array<u32, PrefixDwords> words{};
    u32 total_dwords{};
    u32 copied_dwords{};
    bool has_first_walker_prefix{};
};

inline SrtFailureFlatSnapshot CaptureSrtFailureFlatSnapshot(
    std::span<const u32> flattened) noexcept {
    SrtFailureFlatSnapshot snapshot{
        .total_dwords = static_cast<u32>(flattened.size()),
        .copied_dwords = static_cast<u32>(std::min(
            flattened.size(), static_cast<std::size_t>(SrtFailureFlatSnapshot::PrefixDwords))),
    };
    std::ranges::copy(flattened.first(snapshot.copied_dwords), snapshot.words.begin());
    snapshot.has_first_walker_prefix =
        snapshot.copied_dwords == SrtFailureFlatSnapshot::PrefixDwords;
    return snapshot;
}

} // namespace VideoCore
