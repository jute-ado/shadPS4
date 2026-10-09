// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

#include <atomic>
#include <queue>

namespace Libraries::VideoOut {

// A VideoOutPort can be closed and reopened while GPU work from its previous lifetime is still
// queued. The epoch lets callbacks and requests prove that they belong to the currently open
// lifetime before touching port state.
class VideoOutLifecycle {
public:
    using Epoch = u64;

    [[nodiscard]] Epoch Open() noexcept {
        const auto next_epoch = lifecycle_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
        is_open.store(true, std::memory_order_release);
        return next_epoch;
    }

    void Close() noexcept {
        is_open.store(false, std::memory_order_release);
        lifecycle_epoch.fetch_add(1, std::memory_order_acq_rel);
    }

    [[nodiscard]] bool IsOpen() const noexcept {
        return is_open.load(std::memory_order_acquire);
    }

    [[nodiscard]] Epoch CurrentEpoch() const noexcept {
        return lifecycle_epoch.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool IsCurrent(Epoch epoch) const noexcept {
        return IsOpen() && CurrentEpoch() == epoch;
    }

private:
    std::atomic<Epoch> lifecycle_epoch{};
    std::atomic_bool is_open{false};
};

template <typename Request>
void DrainVideoOutRequests(std::queue<Request>& requests) noexcept {
    std::queue<Request>{}.swap(requests);
}

} // namespace Libraries::VideoOut
