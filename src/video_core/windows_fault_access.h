// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

namespace VideoCore {

enum class WindowsFaultAccess {
    Read,
    Write,
    Execute,
    Unknown,
};

[[nodiscard]] constexpr WindowsFaultAccess DecodeWindowsFaultAccess(std::uintptr_t operation) {
    switch (operation) {
    case 0:
        return WindowsFaultAccess::Read;
    case 1:
        return WindowsFaultAccess::Write;
    case 8:
        return WindowsFaultAccess::Execute;
    default:
        return WindowsFaultAccess::Unknown;
    }
}

[[nodiscard]] bool IsWindowsFaultAccessAllowed(const void* address,
                                               WindowsFaultAccess access) noexcept;

} // namespace VideoCore
