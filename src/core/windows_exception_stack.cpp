// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/windows_exception_stack.h"

#include <cstdint>
#include <memory>

namespace Core {

void* WindowsExceptionStack::TryAcquire() noexcept {
    if (active) {
        return nullptr;
    }
    active = true;
    return storage.data() + storage.size();
}

void WindowsExceptionStack::Release() noexcept {
    active = false;
}

bool WindowsExceptionStack::Contains(const void* pointer) const noexcept {
    const auto value = reinterpret_cast<std::uintptr_t>(pointer);
    const auto begin = reinterpret_cast<std::uintptr_t>(storage.data());
    const auto end = begin + storage.size();
    return value >= begin && value <= end;
}

thread_local std::unique_ptr<WindowsExceptionStack> exception_stack;

void PrepareWindowsExceptionStack() {
    if (!exception_stack) {
        exception_stack = std::make_unique<WindowsExceptionStack>();
    }
}

extern "C" void* AcquireWindowsExceptionStack() noexcept {
    return exception_stack ? exception_stack->TryAcquire() : nullptr;
}

extern "C" void ReleaseWindowsExceptionStack() noexcept {
    if (exception_stack) {
        exception_stack->Release();
    }
}

} // namespace Core
