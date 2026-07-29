// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/windows_memory_protection.h"

#ifdef _WIN32

#include <algorithm>
#include <limits>

namespace Core {
namespace {

[[nodiscard]] bool AllowsExecute(DWORD protection) noexcept {
    switch (protection & 0xff) {
    case PAGE_EXECUTE:
    case PAGE_EXECUTE_READ:
    case PAGE_EXECUTE_READWRITE:
    case PAGE_EXECUTE_WRITECOPY:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] DWORD DataProtection(bool read, bool write, bool execute) noexcept {
    if (execute) {
        if (write) {
            return PAGE_EXECUTE_READWRITE;
        }
        return read ? PAGE_EXECUTE_READ : PAGE_EXECUTE;
    }
    if (write) {
        return PAGE_READWRITE;
    }
    return read ? PAGE_READONLY : PAGE_NOACCESS;
}

} // namespace

bool ProtectWindowsDataAccessPreservingExecute(HANDLE process, VAddr address, u64 size, bool read,
                                               bool write) noexcept {
    if (size == 0) {
        return true;
    }
    if (process == nullptr || address > std::numeric_limits<VAddr>::max() - size) {
        return false;
    }

    const VAddr end = address + size;
    VAddr cursor = address;
    while (cursor < end) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQueryEx(process, reinterpret_cast<const void*>(cursor), &info, sizeof(info)) ==
                0 ||
            info.State != MEM_COMMIT || (info.Protect & PAGE_GUARD) != 0) {
            return false;
        }

        const VAddr region_end =
            reinterpret_cast<VAddr>(info.BaseAddress) + static_cast<u64>(info.RegionSize);
        const VAddr protect_end = std::min(end, region_end);
        if (protect_end <= cursor) {
            return false;
        }

        DWORD old_protection{};
        if (!VirtualProtectEx(process, reinterpret_cast<void*>(cursor), protect_end - cursor,
                              DataProtection(read, write, AllowsExecute(info.Protect)),
                              &old_protection)) {
            return false;
        }
        cursor = protect_end;
    }
    return true;
}

} // namespace Core

#endif
