// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace VideoCore {

template <typename Record>
class DeferredReadyQueue {
public:
    using Consumer = std::function<void(std::span<const Record>)>;

    explicit DeferredReadyQueue(Consumer consume_) : consume{std::move(consume_)} {}

    void Enqueue(std::span<const Record> records) {
        pending.insert(pending.end(), records.begin(), records.end());
    }

    bool Drain() {
        if (draining) {
            return false;
        }
        if (pending.empty()) {
            return true;
        }
        draining = true;
        struct ResetDrain {
            bool& active;
            ~ResetDrain() {
                active = false;
            }
        } reset{draining};
        // A consumer may collect more completions while waiting for an upload. Keep nested
        // drains disabled, but settle those completions at this same safe outer boundary so
        // command preparation cannot continue with ready residency work still pending.
        while (!pending.empty()) {
            std::vector<Record> batch;
            batch.swap(pending);
            consume(batch);
        }
        return true;
    }

private:
    Consumer consume;
    std::vector<Record> pending;
    bool draining{};
};

} // namespace VideoCore
