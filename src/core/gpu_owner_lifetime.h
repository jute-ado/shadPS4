// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

namespace Core {

// Keeps the GPU command owner available for an already-admitted host memory/cache operation.
// Acquisition never depends on guest submissions, GPU completion, or address-range ownership.
class GpuOwnerLifetime {
    struct State {
        std::mutex mutex;
        std::condition_variable idle;
        size_t active{};
        bool closed{};
    };

    struct ActiveOperation {
        explicit ActiveOperation(std::shared_ptr<State> state) : state{std::move(state)} {}

        ~ActiveOperation() {
            {
                std::scoped_lock lock{state->mutex};
                --state->active;
            }
            state->idle.notify_all();
        }

        std::shared_ptr<State> state;
    };

public:
    class Guard {
    public:
        Guard() = default;
        Guard(Guard&&) = default;
        Guard& operator=(Guard&&) = default;

    private:
        explicit Guard(std::unique_ptr<ActiveOperation> operation)
            : operation{std::move(operation)} {}
        std::unique_ptr<ActiveOperation> operation;
        friend class GpuOwnerLifetime;
    };

    [[nodiscard]] std::optional<Guard> Acquire() const {
        std::scoped_lock lock{state->mutex};
        if (state->closed) {
            return std::nullopt;
        }
        auto operation = std::make_unique<ActiveOperation>(state);
        ++state->active;
        return Guard{std::move(operation)};
    }

    // Call before stopping/joining the owner, never from an operation holding its own guard.
    // New users are refused while admitted operations can still request owner-side cache work.
    void CloseAndWait() {
        std::unique_lock lock{state->mutex};
        state->closed = true;
        state->idle.wait(lock, [&] { return state->active == 0; });
    }

private:
    std::shared_ptr<State> state = std::make_shared<State>();
};

} // namespace Core
