// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "core/libraries/kernel/threads/pthread.h"

namespace Libraries::Kernel {

inline void ReleasePthreadAttr(PthreadAttr& attr) noexcept {
    std::free(attr.cpuset);
    attr.cpuset = nullptr;
    attr.cpusetsize = 0;
}

[[nodiscard]] inline bool ClonePthreadAttr(PthreadAttr& destination,
                                           const PthreadAttr& source) noexcept {
    if (&destination == &source) {
        return true;
    }

    Cpuset* cpuset_copy = nullptr;
    if (source.cpuset != nullptr) {
        cpuset_copy = static_cast<Cpuset*>(std::calloc(1, sizeof(Cpuset)));
        if (cpuset_copy == nullptr) {
            return false;
        }
        std::memcpy(cpuset_copy, source.cpuset, std::min(source.cpusetsize, sizeof(Cpuset)));
    }

    ReleasePthreadAttr(destination);
    destination = source;
    destination.cpuset = cpuset_copy;
    destination.cpusetsize = cpuset_copy != nullptr ? sizeof(Cpuset) : 0;
    return true;
}

} // namespace Libraries::Kernel
