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
        // A consumer may collect more completions while waiting for an upload. Those records
        // belong to the next outer boundary, never to this active cache mutation.
        std::vector<Record> batch;
        batch.swap(pending);
        draining = true;
        struct ResetDrain {
            bool& active;
            ~ResetDrain() {
                active = false;
            }
        } reset{draining};
        consume(batch);
        return true;
    }

private:
    Consumer consume;
    std::vector<Record> pending;
    bool draining{};
};

} // namespace VideoCore
