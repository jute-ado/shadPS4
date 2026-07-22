// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>

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

    [[nodiscard]] bool AreSubmitsAllowed(bool gpu_idle) const {
        std::scoped_lock lock{mutex};
        return open && gpu_idle;
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
