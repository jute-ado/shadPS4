// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace VideoCore {

template <typename Job>
class ScreenshotWriterQueue {
public:
    void Start() {
        std::scoped_lock lock{mutex};
        accepting = true;
    }

    bool Push(Job job) {
        {
            std::scoped_lock lock{mutex};
            if (!accepting) {
                return false;
            }
            jobs.push_back(std::move(job));
            ++outstanding;
        }
        ready.notify_one();
        return true;
    }

    bool Reserve() {
        std::scoped_lock lock{mutex};
        if (!accepting) {
            return false;
        }
        ++outstanding;
        return true;
    }

    void PushReserved(Job job) {
        {
            std::scoped_lock lock{mutex};
            jobs.push_back(std::move(job));
        }
        ready.notify_one();
    }

    std::optional<Job> WaitPop() {
        std::unique_lock lock{mutex};
        ready.wait(lock, [this] { return !accepting || !jobs.empty(); });
        if (jobs.empty()) {
            return std::nullopt;
        }
        Job job = std::move(jobs.front());
        jobs.pop_front();
        return job;
    }

    void Complete() {
        {
            std::scoped_lock lock{mutex};
            --outstanding;
        }
        idle.notify_all();
    }

    void WaitIdle() {
        std::unique_lock lock{mutex};
        idle.wait(lock, [this] { return outstanding == 0; });
    }

    void Stop() {
        {
            std::scoped_lock lock{mutex};
            accepting = false;
        }
        ready.notify_all();
    }

private:
    std::mutex mutex;
    std::condition_variable ready;
    std::condition_variable idle;
    std::deque<Job> jobs;
    std::size_t outstanding{};
    bool accepting{};
};

} // namespace VideoCore
