// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <shared_mutex>

namespace Core {

// Mapping and guest-visible protection changes mutate address-space state and must remain
// exclusive. Internal page-tracking protection changes only read that state, and PageManager
// already serializes overlapping page ranges, so independent ranges can proceed concurrently.
class AddressSpaceOperationGate {
public:
    using TrackingLock = std::shared_lock<std::shared_mutex>;
    using MutationLock = std::unique_lock<std::shared_mutex>;

    [[nodiscard]] TrackingLock LockForTracking() {
        return TrackingLock{mutex};
    }

    [[nodiscard]] TrackingLock TryLockForTracking() {
        return TrackingLock{mutex, std::try_to_lock};
    }

    [[nodiscard]] MutationLock LockForMutation() {
        return MutationLock{mutex};
    }

    [[nodiscard]] MutationLock TryLockForMutation() {
        return MutationLock{mutex, std::try_to_lock};
    }

private:
    std::shared_mutex mutex;
};

} // namespace Core
