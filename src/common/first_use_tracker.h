// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>

namespace Common {

template <std::size_t NumValues>
class FirstUseTracker {
public:
    [[nodiscard]] bool IsFirstUse(std::size_t value) noexcept {
        if (value >= NumValues || used[value].test(std::memory_order_relaxed)) {
            return false;
        }
        return !used[value].test_and_set(std::memory_order_relaxed);
    }

private:
    std::array<std::atomic_flag, NumValues> used{};
};

} // namespace Common
