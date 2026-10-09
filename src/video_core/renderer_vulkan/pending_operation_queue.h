// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <concepts>
#include <cstddef>
#include <mutex>
#include <queue>
#include <utility>

#include "common/types.h"

namespace VideoCore {

template <typename Callback>
class PendingOperationQueue {
public:
    void Emplace(Callback&& callback, u64 gpu_tick) {
        std::unique_lock lock{mutex};
        operations.emplace(std::move(callback), gpu_tick);
    }

    template <typename IsReady>
        requires std::predicate<IsReady&, u64>
    void PopReady(IsReady&& is_ready) {
        std::unique_lock lock{mutex};
        while (!operations.empty() && is_ready(operations.front().gpu_tick)) {
            operations.front().callback();
            operations.pop();
        }
    }

    [[nodiscard]] std::size_t Size() const {
        std::unique_lock lock{mutex};
        return operations.size();
    }

private:
    struct Operation {
        Callback callback;
        u64 gpu_tick;
    };

    mutable std::recursive_mutex mutex;
    std::queue<Operation> operations;
};

} // namespace VideoCore
