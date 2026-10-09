// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

#include "common/alignment.h"
#include "common/types.h"

namespace VideoCore {

struct DownloadReservation {
    u64 size;
    u64 alignment;
};

struct DownloadRange {
    VAddr address;
    u64 size;
};

[[nodiscard]] constexpr std::optional<DownloadRange> ClipDownloadRange(
    VAddr tracked_address, u64 tracked_size, VAddr requested_address, u64 requested_size) {
    if (tracked_size > std::numeric_limits<VAddr>::max() - tracked_address ||
        requested_size > std::numeric_limits<VAddr>::max() - requested_address) {
        return std::nullopt;
    }
    const VAddr begin = std::max(tracked_address, requested_address);
    const VAddr end = std::min(tracked_address + tracked_size, requested_address + requested_size);
    if (begin >= end) {
        return std::nullopt;
    }
    return DownloadRange{begin, end - begin};
}

[[nodiscard]] constexpr DownloadReservation PlanDownloadReservation(u64 size, u64 alignment,
                                                                    u64 non_coherent_atom_size,
                                                                    bool is_coherent) {
    if (is_coherent) {
        return {size, alignment};
    }
    return {
        .size = Common::AlignUp(size, non_coherent_atom_size),
        .alignment = std::max(alignment, non_coherent_atom_size),
    };
}

template <typename Finish, typename Invalidate, typename Consume>
bool CompleteDownloadReadback(Finish&& finish, Invalidate&& invalidate, Consume&& consume) {
    std::forward<Finish>(finish)();
    if (!std::forward<Invalidate>(invalidate)()) {
        return false;
    }
    std::forward<Consume>(consume)();
    return true;
}

} // namespace VideoCore

