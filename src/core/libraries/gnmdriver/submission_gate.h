// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <set>

#include "common/unique_function.h"

namespace Libraries::GnmDriver {

class SubmissionGate {
public:
    using Guard = std::unique_lock<std::mutex>;

    struct GuardedBoundary {
        Guard issuance;
        Common::UniqueFunction<void> complete;
    };

    class ScopedBoundary {
    public:
        ScopedBoundary() = default;
        explicit ScopedBoundary(GuardedBoundary boundary_)
            : issuance{std::move(boundary_.issuance)}, complete{std::move(boundary_.complete)} {}

        ScopedBoundary(ScopedBoundary&& other) noexcept
            : issuance{std::move(other.issuance)}, complete{std::move(other.complete)} {}

        ScopedBoundary& operator=(ScopedBoundary&& other) noexcept {
            if (this != &other) {
                Finish();
                issuance = std::move(other.issuance);
                complete = std::move(other.complete);
            }
            return *this;
        }

        ~ScopedBoundary() {
            Finish();
        }

        ScopedBoundary(const ScopedBoundary&) = delete;
        ScopedBoundary& operator=(const ScopedBoundary&) = delete;

    private:
        void Finish() {
            if (issuance.owns_lock()) {
                issuance.unlock();
            }
            if (complete) {
                complete();
                complete = {};
            }
        }

        Guard issuance;
        Common::UniqueFunction<void> complete;
    };

    [[nodiscard]] Guard Enter() {
        return Guard{mutex};
    }

    [[nodiscard]] GuardedBoundary BeginGuardedBoundary() {
        Guard lock{mutex};
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

    [[nodiscard]] ScopedBoundary BeginScopedBoundary() {
        return ScopedBoundary{BeginGuardedBoundary()};
    }

    [[nodiscard]] bool IsBoundaryOpen() const {
        std::scoped_lock lock{mutex};
        return completed_boundary == current_boundary;
    }

    // The gate models host backpressure. Reporting it through sceGnmAreSubmitsAllowed would make
    // the guest discard a frame whose end-of-pipe event cannot be recreated later.
    [[nodiscard]] constexpr bool AreGuestSubmitsAllowed() const noexcept {
        return true;
    }

    void WaitForBoundary() {
        Guard lock{mutex};
        const auto boundary = current_boundary;
        boundary_completed.wait(lock, [this, boundary] { return completed_boundary >= boundary; });
    }

private:
    void CompleteBoundary(std::uint64_t boundary) {
        {
            std::scoped_lock lock{mutex};
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

    mutable std::mutex mutex;
    std::condition_variable boundary_completed;
    std::uint64_t current_boundary{};
    std::uint64_t completed_boundary{};
    std::set<std::uint64_t> completed_out_of_order;
};

} // namespace Libraries::GnmDriver
