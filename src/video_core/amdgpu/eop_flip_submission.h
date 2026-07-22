// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <utility>
#include <vector>

#include "common/unique_function.h"

namespace AmdGpu {

class EopFlipSubmission {
public:
    void AttachFlip(Common::UniqueFunction<void>&& callback) {
        bool run_immediately;
        {
            std::scoped_lock lock{mutex};
            run_immediately = submitted;
            if (!run_immediately) {
                pending_flips.emplace_back(std::move(callback));
            }
        }
        if (run_immediately) {
            callback();
        }
    }

    void CompleteSubmission() {
        std::vector<Common::UniqueFunction<void>> callbacks;
        {
            std::scoped_lock lock{mutex};
            if (submitted) {
                return;
            }
            submitted = true;
            callbacks = std::move(pending_flips);
        }
        for (auto& callback : callbacks) {
            callback();
        }
    }

private:
    std::mutex mutex;
    std::vector<Common::UniqueFunction<void>> pending_flips;
    bool submitted{};
};

} // namespace AmdGpu
