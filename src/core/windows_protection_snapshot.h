// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifdef _WIN32

#include <optional>
#include <span>
#include <vector>
#include <windows.h>

#include "common/types.h"

namespace Core {

struct WindowsProtectionSpan {
    VAddr address;
    u64 size;
    DWORD protection;
};

[[nodiscard]] std::optional<std::vector<WindowsProtectionSpan>> CaptureWindowsProtectionOverrides(
    HANDLE process, VAddr address, u64 size, DWORD default_protection);

[[nodiscard]] bool RestoreWindowsProtectionOverrides(
    HANDLE process, std::span<const WindowsProtectionSpan> protections) noexcept;

} // namespace Core

#endif
