// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Core::WindowsException {

inline constexpr u32 BreakpointExceptionCode = 0x80000003;
inline constexpr u32 MsvcCppExceptionCode = 0xe06d7363;

// Vectored exception handlers run before language-runtime handlers. Returning true here asks the
// emulator to begin shutdown before Windows continues searching for a handler.
constexpr bool ShouldShutdownForUnclaimedException(u32 code) {
    return code != BreakpointExceptionCode && code != MsvcCppExceptionCode;
}

constexpr bool ShouldReportUnhandledException(u32 code, bool static_red_zone_enabled,
                                              bool static_protection_exception) {
    return ShouldShutdownForUnclaimedException(code) &&
           (!static_red_zone_enabled || static_protection_exception);
}

} // namespace Core::WindowsException
