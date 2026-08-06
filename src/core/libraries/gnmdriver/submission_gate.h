// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <utility>

#include "common/unique_function.h"

namespace Libraries::GnmDriver {

class SubmissionGate {
public:
    using Guard = std::unique_lock<std::mutex>;

    [[nodiscard]] Guard Enter() {
        Guard lock{mutex};
        boundary_completed.wait(lock, [this] { return open; });
        return lock;
    }

    [[nodiscard]] Common::UniqueFunction<void> BeginBoundary() {
        Guard lock{mutex};
        boundary_completed.wait(lock, [this] { return open; });
        open = false;
        const auto boundary = ++current_boundary;
        return [this, boundary] { CompleteBoundary(boundary); };
    }

    template <typename SubmitBoundary>
    void SubmitBoundaryAndWait(SubmitBoundary&& submit_boundary) {
        auto complete_boundary = BeginBoundary();
        std::forward<SubmitBoundary>(submit_boundary)(std::move(complete_boundary));
    }

    [[nodiscard]] bool IsBoundaryOpen() const {
        std::scoped_lock lock{mutex};
        return open;
    }

    // The gate models host backpressure. Reporting it through sceGnmAreSubmitsAllowed would make
    // the guest discard a frame whose end-of-pipe event cannot be recreated later.
    [[nodiscard]] constexpr bool AreGuestSubmitsAllowed() const noexcept {
        return true;
    }

    void WaitForBoundary() {
        Guard lock{mutex};
        boundary_completed.wait(lock, [this] { return open; });
    }

private:
    void CompleteBoundary(std::uint64_t boundary) {
        {
            std::scoped_lock lock{mutex};
            if (open || boundary != current_boundary) {
                return;
            }
            open = true;
        }
        boundary_completed.notify_all();
    }

    mutable std::mutex mutex;
    std::condition_variable boundary_completed;
    std::uint64_t current_boundary{};
    bool open{true};
};

} // namespace Libraries::GnmDriver
