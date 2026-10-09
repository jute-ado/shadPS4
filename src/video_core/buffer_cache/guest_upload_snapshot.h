// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <utility>

#include "common/scope_exit.h"

namespace VideoCore {

template <typename Lock, typename Snapshot, typename Unlock>
decltype(auto) WithGuestUploadSnapshot(Lock&& lock, Snapshot&& snapshot, Unlock&& unlock) {
    std::invoke(std::forward<Lock>(lock));
    SCOPE_EXIT {
        std::invoke(std::forward<Unlock>(unlock));
    };
    return std::invoke(std::forward<Snapshot>(snapshot));
}

} // namespace VideoCore
