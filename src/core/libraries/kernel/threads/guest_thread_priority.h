// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/thread.h"
#include "core/libraries/kernel/threads/sched_policy.h"

namespace Libraries::Kernel {

[[nodiscard]] constexpr Common::ThreadPriority MapGuestThreadPriority(SchedPolicy policy,
                                                                       int priority) noexcept {
    if (policy == SchedPolicy::Other) {
        return Common::ThreadPriority::Normal;
    }

    if (priority < ORBIS_KERNEL_PRIO_FIFO_DEFAULT) {
        return Common::ThreadPriority::High;
    }
    if (priority > ORBIS_KERNEL_PRIO_FIFO_DEFAULT) {
        return Common::ThreadPriority::Low;
    }
    return Common::ThreadPriority::Normal;
}

} // namespace Libraries::Kernel
