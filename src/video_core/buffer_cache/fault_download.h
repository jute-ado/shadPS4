// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <optional>

#include "video_core/buffer_cache/dma_publication_gate.h"

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

[[nodiscard]] constexpr DmaFaultEpoch ClassifyDmaFaultEpoch(BoundedFaultDownloadCount count,
                                                            size_t invalid_fault_count) {
    if (count.overflowed) {
        return DmaFaultEpoch::Overflow();
    }
    if (invalid_fault_count != 0) {
        return DmaFaultEpoch::Invalid();
    }
    if (count.address_count != 0) {
        return DmaFaultEpoch::FaultCount(count.address_count);
    }
    return DmaFaultEpoch::Clean();
}

class DmaFaultEpochCompletion {
public:
    constexpr void Reset() {
        epoch.reset();
    }

    constexpr void Complete(DmaFaultEpoch completed_epoch) {
        epoch = completed_epoch;
    }

    [[nodiscard]] constexpr bool IsComplete() const {
        return epoch.has_value();
    }

    [[nodiscard]] constexpr DmaFaultEpoch ValueOrInvalid() const {
        return epoch.value_or(DmaFaultEpoch::Invalid());
    }

private:
    std::optional<DmaFaultEpoch> epoch;
};

} // namespace VideoCore
