// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <utility>
#include <vector>

#include "common/unique_function.h"

namespace AmdGpu {

class EopFlipCompletion {
public:
    void AttachFlip(Common::UniqueFunction<void>&& callback) {
        bool run_immediately;
        {
            std::scoped_lock lock{mutex};
            run_immediately = completed;
            if (!run_immediately) {
                pending_flips.emplace_back(std::move(callback));
            }
        }
        if (run_immediately) {
            callback();
        }
    }

    void CompleteEop() {
        std::vector<Common::UniqueFunction<void>> callbacks;
        {
            std::scoped_lock lock{mutex};
            if (completed) {
                return;
            }
            completed = true;
            callbacks = std::move(pending_flips);
        }
        for (auto& callback : callbacks) {
            callback();
        }
    }

private:
    std::mutex mutex;
    std::vector<Common::UniqueFunction<void>> pending_flips;
    bool completed{};
};

} // namespace AmdGpu
