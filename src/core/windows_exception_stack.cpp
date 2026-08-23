// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/windows_exception_stack.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include <windows.h>

namespace Core {
namespace {

constexpr std::size_t ExceptionStackCommitSize = 128 * 1024;
constexpr std::size_t ExceptionStackReserveSize = 256 * 1024;

struct WindowsExceptionStackState {
    void* prepared_fiber{};
    void* exception_fiber{};
    void* return_fiber{};
    WindowsExceptionStackCallback callback{};
    void* callback_context{};
    std::intptr_t callback_result{};
    bool owns_thread_conversion{};
    bool active{};
    bool cleaning{};
};

struct TebStackBounds {
    void* base{};
    void* limit{};
};

thread_local WindowsExceptionStackState exception_stack_state;

TebStackBounds ReadTebStackBounds() noexcept {
    const auto* tib = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
    return {.base = tib->StackBase, .limit = tib->StackLimit};
}

void RestoreTebStackBounds(TebStackBounds bounds) noexcept {
    auto* tib = reinterpret_cast<NT_TIB*>(NtCurrentTeb());
    tib->StackBase = bounds.base;
    tib->StackLimit = bounds.limit;
}

void WINAPI ExceptionFiberEntry(void* opaque) noexcept {
    auto& state = *static_cast<WindowsExceptionStackState*>(opaque);
    for (;;) {
        const auto callback = state.callback;
        void* const context = state.callback_context;
        state.callback_result = callback != nullptr ? callback(context) : 0;
        SwitchToFiber(state.return_fiber);
    }
}

void DeletePreparedExceptionFiber(WindowsExceptionStackState& state) noexcept {
    state.cleaning = true;
    void* const exception_fiber = std::exchange(state.exception_fiber, nullptr);
    if (exception_fiber != nullptr) {
        // DeleteFiber runs FLS callbacks. Keep the target unpublished so a
        // callback that re-enters exception dispatch takes the direct path
        // instead of switching into the fiber being destroyed.
        DeleteFiber(exception_fiber);
    }
}

} // namespace

bool PrepareWindowsExceptionStack() noexcept {
    auto& state = exception_stack_state;
    if (state.cleaning) {
        return false;
    }
    if (state.exception_fiber != nullptr) {
        if (state.active ||
            (IsThreadAFiber() != FALSE && (GetCurrentFiber() == state.prepared_fiber ||
                                           GetCurrentFiber() == state.exception_fiber))) {
            return true;
        }

        // A borrowed fiber owner can convert back to a thread or switch to a
        // different fiber without going through our cleanup hook. Retire that
        // stale target before preparing against the current execution context.
        DeletePreparedExceptionFiber(state);
        state = {};
    }

    if (state.prepared_fiber == nullptr) {
        const auto bounds = ReadTebStackBounds();
        if (bounds.base == nullptr || bounds.limit == nullptr) {
            return false;
        }

        const bool was_fiber = IsThreadAFiber() != FALSE;
        state.prepared_fiber = was_fiber ? GetCurrentFiber()
                                         : ConvertThreadToFiberEx(nullptr, FIBER_FLAG_FLOAT_SWITCH);
        if (state.prepared_fiber == nullptr) {
            return false;
        }
        state.owns_thread_conversion = !was_fiber;
    } else if (IsThreadAFiber() == FALSE || GetCurrentFiber() != state.prepared_fiber) {
        return false;
    }

    void* const exception_fiber =
        CreateFiberEx(ExceptionStackCommitSize, ExceptionStackReserveSize, FIBER_FLAG_FLOAT_SWITCH,
                      ExceptionFiberEntry, &state);
    if (exception_fiber == nullptr) {
        if (!state.owns_thread_conversion || ConvertFiberToThread() != FALSE) {
            state = {};
        }
        return false;
    }

    state.exception_fiber = exception_fiber;
    return true;
}

void CleanupWindowsExceptionStack() noexcept {
    auto& state = exception_stack_state;
    if (state.prepared_fiber == nullptr || state.active || state.cleaning ||
        IsThreadAFiber() == FALSE) {
        return;
    }
    if (state.exception_fiber != nullptr && GetCurrentFiber() == state.exception_fiber) {
        return;
    }
    if (state.owns_thread_conversion && GetCurrentFiber() != state.prepared_fiber) {
        return;
    }

    DeletePreparedExceptionFiber(state);
    const bool owns_thread_conversion = state.owns_thread_conversion;
    if (!owns_thread_conversion || ConvertFiberToThread() != FALSE) {
        state = {};
    } else {
        state.cleaning = false;
    }
}

void ReleaseWindowsExceptionStackForThreadExit() noexcept {
    auto& state = exception_stack_state;
    if (state.prepared_fiber == nullptr || state.active || state.cleaning ||
        (state.exception_fiber != nullptr && IsThreadAFiber() != FALSE &&
         GetCurrentFiber() == state.exception_fiber)) {
        return;
    }

    DeletePreparedExceptionFiber(state);
    // ExitThread owns final teardown of the current converted fiber. Avoid
    // ConvertFiberToThread here because a guest can exit while its stack
    // pointer is temporarily on a manually managed guest stack.
    state = {};
}

std::intptr_t RunOnWindowsExceptionStack(WindowsExceptionStackCallback callback,
                                         void* context) noexcept {
    if (callback == nullptr) {
        return 0;
    }

    auto& state = exception_stack_state;
    if (state.exception_fiber == nullptr || state.cleaning || IsThreadAFiber() == FALSE) {
        return callback(context);
    }

    // A second exception while the handler body is active already has the
    // prepared stack. Re-enter directly so the outer callback state remains
    // intact and the same stack is used for nested delivery.
    if (state.active || GetCurrentFiber() == state.exception_fiber) {
        return callback(context);
    }

    state.return_fiber = GetCurrentFiber();
    state.callback = callback;
    state.callback_context = context;
    state.callback_result = 0;
    state.active = true;
    const auto teb_stack_bounds = ReadTebStackBounds();
    SwitchToFiber(state.exception_fiber);
    RestoreTebStackBounds(teb_stack_bounds);
    state.active = false;

    const auto result = state.callback_result;
    state.return_fiber = nullptr;
    state.callback = nullptr;
    state.callback_context = nullptr;
    return result;
}

bool IsOnPreparedWindowsExceptionStack(const void* address) noexcept {
    const auto& state = exception_stack_state;
    if (address == nullptr || state.exception_fiber == nullptr || !state.active ||
        IsThreadAFiber() == FALSE || GetCurrentFiber() != state.exception_fiber) {
        return false;
    }

    ULONG_PTR stack_low{};
    ULONG_PTR stack_high{};
    GetCurrentThreadStackLimits(&stack_low, &stack_high);
    const auto value = reinterpret_cast<std::uintptr_t>(address);
    return value >= stack_low && value < stack_high;
}

} // namespace Core
