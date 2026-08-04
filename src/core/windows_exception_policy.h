// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>

#include "common/types.h"

namespace Core::WindowsException {

inline constexpr u32 BreakpointExceptionCode = 0x80000003;
inline constexpr u32 MsvcCppExceptionCode = 0xe06d7363;

// Vectored exception handlers run before language-runtime handlers. Returning true here asks the
// emulator to begin shutdown before Windows continues searching for a handler.
constexpr bool ShouldShutdownForUnclaimedException(u32 code) {
    return code != BreakpointExceptionCode && code != MsvcCppExceptionCode;
}

constexpr std::string_view AccessViolationOperationName(u64 operation) {
    switch (operation) {
    case 0:
        return "read";
    case 1:
        return "write";
    case 8:
        return "execute";
    default:
        return "unknown";
    }
}

} // namespace Core::WindowsException
