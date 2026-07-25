// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/kernel/equeue.h"
#include "core/libraries/kernel/orbis_error.h"

namespace Libraries::Kernel {

inline int ValidateEqueueWaitArguments(OrbisKernelEvent* events, int capacity, int* count) {
    if (events == nullptr || count == nullptr) {
        return ORBIS_KERNEL_ERROR_EFAULT;
    }
    if (capacity < 1) {
        *count = 0;
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    return ORBIS_OK;
}

} // namespace Libraries::Kernel
