// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <utility>

namespace Libraries::Kernel {

// Zero timeout skips waiting for a new event, not synchronization with current producers.
// Collection copies/clears events and may erase one-shot entries. Blocking waits already
// retain this mutex while collecting and must not call this helper recursively.
template <typename Collect>
decltype(auto) PollEventQueue(std::mutex& mutex, Collect&& collect) {
    std::scoped_lock lock{mutex};
    return std::forward<Collect>(collect)();
}

} // namespace Libraries::Kernel
