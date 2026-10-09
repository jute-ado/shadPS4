// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <optional>

namespace Vulkan {

// Diagnostic only: no guest clock, waits, or synchronization. Disabled diagnostics
// do not sample the clock. A fake steady clock keeps unit tests independent of sleep.
template <typename Clock = std::chrono::steady_clock>
class CompilationLatencyTimer {
    static_assert(Clock::is_steady);

public:
    explicit CompilationLatencyTimer(bool enabled) {
        if (enabled) {
            started = Clock::now();
        }
    }

    [[nodiscard]] bool IsEnabled() const noexcept {
        return started.has_value();
    }

    [[nodiscard]] double ElapsedMilliseconds() const {
        if (!started) {
            return 0.0;
        }
        return std::chrono::duration<double, std::milli>(Clock::now() - *started).count();
    }

private:
    std::optional<typename Clock::time_point> started;
};

} // namespace Vulkan
