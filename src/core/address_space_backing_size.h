// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <optional>

#include "common/types.h"
#include "core/libraries/kernel/memory.h"

namespace Core {

inline constexpr u64 AddressSpaceBaseBackingSize =
    ORBIS_KERNEL_TOTAL_MEM_DEV_PRO + ORBIS_KERNEL_FLEXIBLE_MEMORY_SIZE;

[[nodiscard]] constexpr std::optional<u64> CalculateAddressSpaceBackingSize(
    s64 extra_dmem_in_mbytes) noexcept {
    if (extra_dmem_in_mbytes < 0) {
        return std::nullopt;
    }

    constexpr u64 MByte = 1_MB;
    const auto extra_dmem = static_cast<u64>(extra_dmem_in_mbytes);
    if (extra_dmem > (std::numeric_limits<u64>::max() - AddressSpaceBaseBackingSize) / MByte) {
        return std::nullopt;
    }
    return AddressSpaceBaseBackingSize + extra_dmem * MByte;
}

} // namespace Core
