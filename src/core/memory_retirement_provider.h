// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include "common/polyfill_thread.h"
#include "common/unique_function.h"

namespace Core {

// An owner-installed lifetime boundary, separate from submission admission. A fresh mapping
// needs only Lease(); only replacement or retirement waits for old command/address ownership.
class MemoryRetirementProvider {
public:
    // The token owns an RAII guard in its closure. Destruction releases admission; it is never
    // invoked as a release callback, including during exception or cancellation unwinding.
    using Token = Common::UniqueFunction<void>;
    using Acquire = Common::UniqueFunction<std::optional<Token>, std::stop_token>;

private:
    struct State {
        explicit State(Acquire acquire) : acquire{std::move(acquire)} {}

        std::mutex mutex;
        std::condition_variable idle;
        std::stop_source stop;
        Acquire acquire;
        size_t active{};
        bool closed{};
    };

    struct LeaseState {
        explicit LeaseState(std::shared_ptr<State> state) : state{std::move(state)} {}
        ~LeaseState() {
            // Release submission admission before acknowledging owner teardown.
            token = {};
            {
                std::scoped_lock lock{state->mutex};
                --state->active;
            }
            state->idle.notify_all();
        }
        std::shared_ptr<State> state;
        Token token;
    };

public:
    class Admission {
    public:
        Admission() = default;
        Admission(Admission&&) = default;
        Admission& operator=(Admission&&) = default;

    private:
        explicit Admission(std::unique_ptr<LeaseState> lease) : lease{std::move(lease)} {}
        std::unique_ptr<LeaseState> lease;
        friend class MemoryRetirementProvider;
    };

    explicit MemoryRetirementProvider(Acquire acquire)
        : state{std::make_shared<State>(std::move(acquire))} {}

    [[nodiscard]] std::optional<Admission> Lease() {
        std::scoped_lock lock{state->mutex};
        if (state->closed) {
            return std::nullopt;
        }
        auto lease = std::make_unique<LeaseState>(state);
        ++state->active;
        return Admission{std::move(lease)};
    }

    [[nodiscard]] std::optional<Admission> Retire() {
        auto admission = Lease();
        if (!admission) {
            return std::nullopt;
        }
        // No lifetime mutex is held over GPU-owner work. Exceptions retain their original
        // meaning; only an explicit owner cancellation is represented by an empty result.
        auto token = state->acquire(state->stop.get_token());
        if (!token || state->stop.stop_requested()) {
            return std::nullopt;
        }
        admission->lease->token = std::move(*token);
        return admission;
    }

    // Called before stopping/joining the GPU owner, never recursively from an admission.
    void CloseAndWait() {
        {
            std::scoped_lock lock{state->mutex};
            state->closed = true;
        }
        state->stop.request_stop();
        std::unique_lock lock{state->mutex};
        state->idle.wait(lock, [&] { return state->active == 0; });
    }

private:
    std::shared_ptr<State> state;
};

} // namespace Core
