// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <set>

#include "common/polyfill_thread.h"
#include "common/unique_function.h"

namespace Libraries::GnmDriver {

class SubmissionGate {
public:
    using Guard = std::unique_lock<std::mutex>;

    struct GuardedBoundary {
        Guard issuance;
        Common::UniqueFunction<void> complete;
    };

    [[nodiscard]] Guard Enter() {
        return Guard{admission_mutex};
    }

    [[nodiscard]] GuardedBoundary BeginGuardedBoundary() {
        Guard lock{admission_mutex};
        std::scoped_lock state_lock{state_mutex};
        const auto boundary = ++current_boundary;
        return {
            .issuance = std::move(lock),
            .complete = [this, boundary] { CompleteBoundary(boundary); },
        };
    }

    [[nodiscard]] Common::UniqueFunction<void> BeginBoundary() {
        auto boundary = BeginGuardedBoundary();
        return std::move(boundary.complete);
    }

    [[nodiscard]] bool IsBoundaryOpen() const {
        std::scoped_lock lock{state_mutex};
        return completed_boundary == current_boundary;
    }

    // The gate models host backpressure. Reporting it through sceGnmAreSubmitsAllowed would make
    // the guest discard a frame whose end-of-pipe event cannot be recreated later.
    [[nodiscard]] constexpr bool AreGuestSubmitsAllowed() const noexcept {
        return true;
    }

    void WaitForBoundary() {
        std::unique_lock lock{state_mutex};
        const auto boundary = current_boundary;
        boundary_completed.wait(lock, [this, boundary] { return completed_boundary >= boundary; });
    }

    [[nodiscard]] bool WaitForBoundary(std::stop_token stop) {
        std::unique_lock lock{state_mutex};
        const auto boundary = current_boundary;
        Common::CondvarWait(boundary_completed, lock, stop,
                            [this, boundary] { return completed_boundary >= boundary; });
        return !stop.stop_requested();
    }

    // Unlike the ordinary captured-boundary wait, retirement may retain admission only when every
    // currently published callback has acknowledged. This keeps a new boundary from being
    // published while retirement synchronously requests owner readback/Finish.
    [[nodiscard]] std::optional<Guard> TryEnterRetirement() {
        Guard lock{admission_mutex, std::try_to_lock};
        if (!lock) {
            return std::nullopt;
        }
        std::scoped_lock state_lock{state_mutex};
        if (completed_boundary != current_boundary) {
            return std::nullopt;
        }
        return lock;
    }

    /// Acquires submission admission before observing or waiting for the current boundary. The
    /// guard remains held through the wait and is returned to the caller for its subsequent
    /// mutation, so later guest issuance cannot overtake an address-space change.
    [[nodiscard]] Guard WaitForBoundaryAndEnter() {
        Guard admission{admission_mutex};
        std::unique_lock state_lock{state_mutex};
        const auto boundary = current_boundary;
        boundary_completed.wait(state_lock,
                                [this, boundary] { return completed_boundary >= boundary; });
        return admission;
    }

    /// Diagnostic-only variant that permits independent guest submissions to make parser
    /// progress while waiting. Address-space changes must use WaitForBoundaryAndEnter(), which
    /// retains admission for the complete wait and caller-owned mutation.
    [[nodiscard]] Guard WaitForBoundaryThenEnter() {
        WaitForBoundary();
        return Enter();
    }

private:
    void CompleteBoundary(std::uint64_t boundary) {
        {
            std::scoped_lock lock{state_mutex};
            if (boundary <= completed_boundary || boundary > current_boundary) {
                return;
            }
            completed_out_of_order.insert(boundary);
            while (completed_out_of_order.erase(completed_boundary + 1) != 0) {
                ++completed_boundary;
            }
        }
        boundary_completed.notify_all();
    }

    // Admission serializes guest issuance and address-space revocation. Completion callbacks
    // must not need this mutex: a revocation can synchronously wait for owner readback while
    // retaining admission, and the owner may concurrently acknowledge a SubmitDone boundary.
    mutable std::mutex admission_mutex;
    mutable std::mutex state_mutex;
    std::condition_variable_any boundary_completed;
    std::uint64_t current_boundary{};
    std::uint64_t completed_boundary{};
    std::set<std::uint64_t> completed_out_of_order;
};

} // namespace Libraries::GnmDriver
