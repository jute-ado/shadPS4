// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Core {

constexpr VAddr MaxGpuVirtualAddress = 0x10000000000ULL;

[[nodiscard]] constexpr bool IsGpuAddressRangeValid(const VAddr address, const u64 size) {
    return address < MaxGpuVirtualAddress && size <= MaxGpuVirtualAddress - address;
}

} // namespace Core
