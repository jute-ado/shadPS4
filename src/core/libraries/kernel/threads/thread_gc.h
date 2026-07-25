// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstddef>

namespace Libraries::Kernel {

class ThreadGcAccounting {
public:
    void MarkQueued() noexcept {
        pending.fetch_add(1);
    }

    void MarkCollected() noexcept {
        pending.fetch_sub(1);
    }

    [[nodiscard]] bool NeedsCollection(const std::size_t threshold) const noexcept {
        return PendingCount() >= threshold;
    }

    [[nodiscard]] std::size_t PendingCount() const noexcept {
        return pending.load();
    }

private:
    std::atomic_size_t pending{};
};

} // namespace Libraries::Kernel
