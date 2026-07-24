// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Libraries::Kernel {

enum class EqueueWaitSource {
    RegularEvent,
    SmallTimer,
    ConditionVariable,
};

constexpr EqueueWaitSource SelectEqueueWaitSource(bool has_ready_regular_event,
                                                  bool has_small_timer) {
    if (has_ready_regular_event) {
        return EqueueWaitSource::RegularEvent;
    }
    if (has_small_timer) {
        return EqueueWaitSource::SmallTimer;
    }
    return EqueueWaitSource::ConditionVariable;
}

} // namespace Libraries::Kernel
