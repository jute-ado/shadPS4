// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/thread.h"
#include "core/libraries/kernel/threads/sched_policy.h"

namespace Libraries::Kernel {

[[nodiscard]] constexpr Common::ThreadPriority MapGuestThreadPriority(SchedPolicy policy,
                                                                       int priority) noexcept {
    return Common::ThreadPriority::Normal;
}

} // namespace Libraries::Kernel
