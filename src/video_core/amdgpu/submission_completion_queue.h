// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include "common/unique_function.h"

namespace AmdGpu {

class SubmissionCompletionQueue final {
public:
    using Sequence = std::uint64_t;
    using Callback = Common::UniqueFunction<void>;

    [[nodiscard]] Sequence IssueSubmission() {
        std::scoped_lock lock{mutex};
        return ++issued_sequence;
    }

    void EnqueueBoundary(Callback&& callback) {
        std::scoped_lock lock{mutex};
        boundaries.push_back({issued_sequence, std::move(callback)});
    }

    void CompleteSubmission(Sequence sequence) {
        std::vector<Callback> ready;
        {
            std::scoped_lock lock{mutex};
            if (sequence == 0 || sequence > issued_sequence || sequence <= completed_sequence) {
                return;
            }
            completed_out_of_order.insert(sequence);
            while (completed_out_of_order.erase(completed_sequence + 1) != 0) {
                ++completed_sequence;
            }
            TakeReadyBoundariesLocked(ready);
        }
        RunReadyBoundaries(ready);
    }

    void DrainReadyBoundaries() {
        std::vector<Callback> ready;
        {
            std::scoped_lock lock{mutex};
            TakeReadyBoundariesLocked(ready);
        }
        RunReadyBoundaries(ready);
    }

    [[nodiscard]] bool HasPendingBoundaries() const {
        std::scoped_lock lock{mutex};
        return !boundaries.empty();
    }

private:
    struct Boundary {
        Sequence target_sequence{};
        Callback callback{};
    };

    void TakeReadyBoundariesLocked(std::vector<Callback>& ready) {
        while (!boundaries.empty() && boundaries.front().target_sequence <= completed_sequence) {
            ready.push_back(std::move(boundaries.front().callback));
            boundaries.pop_front();
        }
    }

    static void RunReadyBoundaries(std::vector<Callback>& ready) {
        for (auto& callback : ready) {
            if (callback) {
                callback();
            }
        }
    }

    mutable std::mutex mutex;
    Sequence issued_sequence{};
    Sequence completed_sequence{};
    std::set<Sequence> completed_out_of_order;
    std::deque<Boundary> boundaries;
};

} // namespace AmdGpu
