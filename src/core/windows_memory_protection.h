// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifdef _WIN32

#include <windows.h>

#include "common/types.h"

namespace Core {

[[nodiscard]] bool ProtectWindowsDataAccessPreservingExecute(HANDLE process, VAddr address,
                                                             u64 size, bool read,
                                                             bool write) noexcept;

} // namespace Core

#endif
