// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>

namespace VideoCore {

inline constexpr size_t FaultDownloadSlotCount = 1024;

struct BoundedFaultDownloadCount {
    size_t address_count;
    bool overflowed;
};

[[nodiscard]] constexpr BoundedFaultDownloadCount BoundFaultDownloadCount(
    size_t reported_count, size_t download_slot_count) {
    // Slot zero stores the counter. Only the remaining slots contain addresses.
    const size_t address_capacity = download_slot_count == 0 ? 0 : download_slot_count - 1;
    return {
        .address_count = reported_count < address_capacity ? reported_count : address_capacity,
        .overflowed = reported_count > address_capacity,
    };
}

} // namespace VideoCore
