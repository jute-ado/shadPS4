// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <shared_mutex>

namespace Core {

template <typename Mutex>
std::shared_lock<Mutex> TryAcquireSharedLock(Mutex& mutex) {
    return std::shared_lock<Mutex>{mutex, std::try_to_lock};
}

} // namespace Core
