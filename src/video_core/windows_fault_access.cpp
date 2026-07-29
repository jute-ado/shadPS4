// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/windows_fault_access.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace VideoCore {

bool IsWindowsFaultAccessAllowed(const void* address, WindowsFaultAccess access) noexcept {
#ifdef _WIN32
    if (address == nullptr || access == WindowsFaultAccess::Execute ||
        access == WindowsFaultAccess::Unknown) {
        return false;
    }

    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(address, &info, sizeof(info)) == 0 || info.State != MEM_COMMIT ||
        (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)) != 0) {
        return false;
    }

    switch (info.Protect & 0xff) {
    case PAGE_READWRITE:
    case PAGE_WRITECOPY:
    case PAGE_EXECUTE_READWRITE:
    case PAGE_EXECUTE_WRITECOPY:
        return true;
    case PAGE_READONLY:
    case PAGE_EXECUTE_READ:
        return access == WindowsFaultAccess::Read;
    default:
        return false;
    }
#else
    (void)address;
    (void)access;
    return false;
#endif
}

} // namespace VideoCore
