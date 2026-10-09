// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

namespace Core {

using WindowsExceptionStackCallback = std::intptr_t (*)(void*) noexcept;

// Prepares an OS-managed alternate stack for the current native thread.
// Failure is non-fatal: RunOnWindowsExceptionStack preserves the old direct
// handler path when no stack has been prepared.
[[nodiscard]] bool PrepareWindowsExceptionStack() noexcept;
void CleanupWindowsExceptionStack() noexcept;
// Deletes only the alternate handler fiber. ExitThread releases the current
// converted fiber without requiring a conversion from a manual guest stack.
void ReleaseWindowsExceptionStackForThreadExit() noexcept;

[[nodiscard]] std::intptr_t RunOnWindowsExceptionStack(WindowsExceptionStackCallback callback,
                                                       void* context) noexcept;

// Diagnostic seam used by the focused stack-pivot contract.
[[nodiscard]] bool IsOnPreparedWindowsExceptionStack(const void* address) noexcept;

} // namespace Core
