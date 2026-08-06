// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>
#include <vector>

namespace Vulkan {

template <typename Queue, typename Ready>
[[nodiscard]] auto TakeReadyPendingOperations(Queue& pending, Ready&& is_ready)
    -> std::vector<typename Queue::value_type> {
    std::vector<typename Queue::value_type> ready;
    while (!pending.empty() && is_ready(pending.front())) {
        ready.push_back(std::move(pending.front()));
        pending.pop();
    }
    return ready;
}

template <typename Queue, typename Ready, typename Execute>
void DrainReadyPendingOperations(Queue& pending, bool& is_draining, Ready&& is_ready,
                                 Execute&& execute) {
    if (std::exchange(is_draining, true)) {
        return;
    }
    struct DrainGuard {
        bool& is_draining;
        ~DrainGuard() {
            is_draining = false;
        }
    } guard{is_draining};

    while (true) {
        auto ready = TakeReadyPendingOperations(pending, is_ready);
        if (ready.empty()) {
            return;
        }
        for (auto& operation : ready) {
            execute(operation);
        }
    }
}

} // namespace Vulkan
