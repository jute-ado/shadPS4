// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Libraries::Kernel {

enum class CondWaitDecision {
    Woken,
    Canceled,
    TimedOut,
    Retry,
};

[[nodiscard]] constexpr CondWaitDecision DecideCondWait(bool is_still_queued,
                                                        bool should_cancel, bool timed_out) {
    if (!is_still_queued) {
        return CondWaitDecision::Woken;
    }
    if (should_cancel) {
        return CondWaitDecision::Canceled;
    }
    if (timed_out) {
        return CondWaitDecision::TimedOut;
    }
    return CondWaitDecision::Retry;
}

} // namespace Libraries::Kernel
