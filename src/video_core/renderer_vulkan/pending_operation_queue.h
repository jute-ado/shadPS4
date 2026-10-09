// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <mutex>
#include <queue>
#include <utility>

namespace Vulkan {

// A FIFO used for GPU-completion callbacks. Callbacks may enqueue follow-up work, so enqueue and
// drain share a recursive lock. A nested drain is ignored; the outer drain observes newly queued
// ready work after the current callback returns.
template <typename Operation>
class PendingOperationQueue {
public:
    void Push(Operation operation) {
        std::scoped_lock lock{mutex};
        operations.push(std::move(operation));
    }

    template <typename IsReady, typename Execute>
    std::size_t DrainReady(IsReady&& is_ready, Execute&& execute) {
        std::scoped_lock lock{mutex};
        if (is_draining) {
            return 0;
        }

        struct DrainGuard {
            bool& value;
            ~DrainGuard() {
                value = false;
            }
        } guard{is_draining};
        is_draining = true;

        std::size_t drained{};
        while (!operations.empty() && is_ready(operations.front())) {
            auto operation = std::move(operations.front());
            operations.pop();
            execute(std::move(operation));
            ++drained;
        }
        return drained;
    }

    [[nodiscard]] std::size_t Size() const {
        std::scoped_lock lock{mutex};
        return operations.size();
    }

    [[nodiscard]] bool Empty() const {
        std::scoped_lock lock{mutex};
        return operations.empty();
    }

private:
    mutable std::recursive_mutex mutex;
    std::queue<Operation> operations;
    bool is_draining{};
};

} // namespace Vulkan
