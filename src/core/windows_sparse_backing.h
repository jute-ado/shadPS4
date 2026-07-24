// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifdef _WIN32
#include <windows.h>
#include "common/types.h"

namespace Core::WindowsSparseBacking {

inline constexpr u32 SectionAllocationType = SEC_RESERVE;
inline constexpr u64 SequentialMappingSize = 1024 * 1024;
inline constexpr u64 MaxPreSplitCount = 64;

[[nodiscard]] constexpr u64 PlaceholderPreSplitCount(u64 mapping_size, u64 available_size) {
    if (mapping_size != SequentialMappingSize || available_size < mapping_size) {
        return 1;
    }
    const u64 available_count = available_size / mapping_size;
    return available_count < MaxPreSplitCount ? available_count : MaxPreSplitCount;
}

[[nodiscard]] constexpr bool PlaceholderNeedsCoalesce(bool is_mapped, u64 available_size,
                                                      u64 mapping_size) {
    return !is_mapped && available_size < mapping_size;
}

[[nodiscard]] inline void* CommitRange(void* base, u64 offset, u64 size, u32 protection) {
    return VirtualAlloc(static_cast<u8*>(base) + offset, size, MEM_COMMIT, protection);
}

} // namespace Core::WindowsSparseBacking
#endif
