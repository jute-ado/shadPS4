// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <shared_mutex>
#include <utility>

namespace Common {

template <typename Mutex, typename Validate, typename Access>
bool WithValidatedSharedAccess(Mutex& mutex, Validate&& validate, Access&& access) {
    {
        std::shared_lock lock{mutex};
        if (!std::forward<Validate>(validate)()) {
            return false;
        }
    }
    std::forward<Access>(access)();
    return true;
}

} // namespace Common
