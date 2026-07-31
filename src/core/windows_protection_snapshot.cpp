// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/windows_protection_snapshot.h"

#ifdef _WIN32

namespace Core {

std::optional<std::vector<WindowsProtectionSpan>> CaptureWindowsProtectionOverrides(
    HANDLE process, VAddr address, u64 size, DWORD default_protection) noexcept {
    static_cast<void>(process);
    static_cast<void>(address);
    static_cast<void>(size);
    static_cast<void>(default_protection);
    return std::nullopt;
}

bool RestoreWindowsProtectionOverrides(
    HANDLE process, std::span<const WindowsProtectionSpan> protections) noexcept {
    static_cast<void>(process);
    static_cast<void>(protections);
    return false;
}

} // namespace Core

#endif
