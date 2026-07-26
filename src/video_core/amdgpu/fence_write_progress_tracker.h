// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <unordered_map>

#include "common/types.h"

namespace AmdGpu {

class FenceWriteProgressTracker {
public:
    struct Progress {
        u64 scheduled{};
        u64 started{};
        u64 completed{};
    };

    void Schedule(VAddr address) {
        std::scoped_lock lock{mutex};
        ++progress_by_address[address].scheduled;
    }

    void Complete(VAddr address) {
        std::scoped_lock lock{mutex};
        ++progress_by_address[address].completed;
    }

    void Start(VAddr address) {
        std::scoped_lock lock{mutex};
        ++progress_by_address[address].started;
    }

    [[nodiscard]] Progress Snapshot(VAddr address) const {
        std::scoped_lock lock{mutex};
        const auto it = progress_by_address.find(address);
        return it == progress_by_address.end() ? Progress{} : it->second;
    }

private:
    mutable std::mutex mutex;
    std::unordered_map<VAddr, Progress> progress_by_address;
};

} // namespace AmdGpu
