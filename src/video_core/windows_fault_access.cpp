// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/windows_fault_access.h"

namespace VideoCore {

bool IsWindowsFaultAccessAllowed(const void* address, WindowsFaultAccess access) noexcept {
    (void)address;
    (void)access;
    return false;
}

} // namespace VideoCore
