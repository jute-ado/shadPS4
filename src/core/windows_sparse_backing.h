// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifdef _WIN32
#include <windows.h>
#include "common/types.h"

namespace Core::WindowsSparseBacking {

inline constexpr u32 SectionAllocationType = SEC_RESERVE;

[[nodiscard]] inline void* CommitRange(void* base, u64 offset, u64 size, u32 protection) {
    return VirtualAlloc(static_cast<u8*>(base) + offset, size, MEM_COMMIT, protection);
}

} // namespace Core::WindowsSparseBacking
#endif
