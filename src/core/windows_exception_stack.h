// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>

namespace Core {

class WindowsExceptionStack {
public:
    static constexpr std::size_t StackSize = 128 * 1024;

    [[nodiscard]] void* TryAcquire() noexcept;
    void Release() noexcept;

    [[nodiscard]] bool IsActive() const noexcept {
        return active;
    }

    [[nodiscard]] bool Contains(const void* pointer) const noexcept;

private:
    alignas(16) std::array<std::byte, StackSize> storage{};
    bool active{};
};

extern "C" void* AcquireWindowsExceptionStack() noexcept;
extern "C" void ReleaseWindowsExceptionStack() noexcept;
void PrepareWindowsExceptionStack();

} // namespace Core
