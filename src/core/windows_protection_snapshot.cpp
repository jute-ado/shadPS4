// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/windows_protection_snapshot.h"

#ifdef _WIN32

#include <algorithm>
#include <limits>

namespace Core {

std::optional<std::vector<WindowsProtectionSpan>> CaptureWindowsProtectionOverrides(
    HANDLE process, VAddr address, u64 size, DWORD default_protection) {
    if (process == nullptr || address > std::numeric_limits<VAddr>::max() - size) {
        return std::nullopt;
    }

    std::vector<WindowsProtectionSpan> protections;
    const VAddr end = address + size;
    for (VAddr cursor = address; cursor < end;) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQueryEx(process, reinterpret_cast<const void*>(cursor), &info, sizeof(info)) ==
                0 ||
            info.State != MEM_COMMIT) {
            return std::nullopt;
        }

        const VAddr region_address = reinterpret_cast<VAddr>(info.BaseAddress);
        if (region_address > std::numeric_limits<VAddr>::max() - info.RegionSize) {
            return std::nullopt;
        }
        const VAddr protection_end = std::min(end, region_address + info.RegionSize);
        if (protection_end <= cursor) {
            return std::nullopt;
        }

        if (info.Protect != default_protection) {
            if (!protections.empty() &&
                protections.back().address + protections.back().size == cursor &&
                protections.back().protection == info.Protect) {
                protections.back().size += protection_end - cursor;
            } else {
                protections.push_back(
                    WindowsProtectionSpan{cursor, protection_end - cursor, info.Protect});
            }
        }
        cursor = protection_end;
    }
    return protections;
}

bool RestoreWindowsProtectionOverrides(
    HANDLE process, std::span<const WindowsProtectionSpan> protections) noexcept {
    if (process == nullptr) {
        return false;
    }

    for (const auto& protection : protections) {
        if (protection.address > std::numeric_limits<VAddr>::max() - protection.size) {
            return false;
        }

        const VAddr end = protection.address + protection.size;
        for (VAddr cursor = protection.address; cursor < end;) {
            MEMORY_BASIC_INFORMATION info{};
            if (VirtualQueryEx(process, reinterpret_cast<const void*>(cursor), &info,
                               sizeof(info)) == 0 ||
                info.State != MEM_COMMIT) {
                return false;
            }

            const VAddr region_address = reinterpret_cast<VAddr>(info.BaseAddress);
            if (region_address > std::numeric_limits<VAddr>::max() - info.RegionSize) {
                return false;
            }
            const VAddr restore_end = std::min(end, region_address + info.RegionSize);
            if (restore_end <= cursor) {
                return false;
            }

            if (info.Protect != protection.protection) {
                DWORD old_protection{};
                if (!VirtualProtectEx(process, reinterpret_cast<void*>(cursor),
                                      restore_end - cursor, protection.protection,
                                      &old_protection)) {
                    return false;
                }
            }
            cursor = restore_end;
        }
    }
    return true;
}

} // namespace Core

#endif
