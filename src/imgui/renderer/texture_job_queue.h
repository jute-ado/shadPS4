// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace ImGui::Core::TextureManager {

template <typename Job>
class JobQueue {
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
        }
        ready.notify_one();
        return true;
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
    std::deque<Job> jobs;
    bool accepting = false;
};

} // namespace ImGui::Core::TextureManager
