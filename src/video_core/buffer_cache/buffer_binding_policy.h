// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

template <typename IsMapped>
[[nodiscard]] bool IsGuestBufferBindingUsable(VAddr address, u64 declared_size,
                                               IsMapped&& is_mapped) {
    constexpr VAddr MaxGpuAddress = 1ULL << 40;
    if (address == 0 || declared_size == 0 || address >= MaxGpuAddress ||
        declared_size > MaxGpuAddress - address) {
        return false;
    }

    // Large and sparse descriptors are legal. Admission only proves that the descriptor base is
    // guest GPU memory; MemoryManager::ClampRangeSize handles its usable tail afterward.
    return is_mapped(address, 1);
}

} // namespace VideoCore
