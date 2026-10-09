// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_set>
#include <utility>

namespace Vulkan {

// A single-worker queue for host compilation. The owner submits immutable jobs and publishes
// completed results on its own thread. Keeping publication outside the worker prevents cache maps
// and renderer-visible pointers from becoming concurrently mutable.
template <typename Key, typename Result, typename Job, typename Hash = std::hash<Key>>
class AsyncCompilationQueue final {
public:
    struct Completion {
        Key key;
        std::optional<Result> result;
    };

    using Compiler = std::function<Result(Job)>;

    explicit AsyncCompilationQueue(Compiler compiler_)
        : compiler{std::move(compiler_)}, worker{[this] { WorkerLoop(); }} {}

    ~AsyncCompilationQueue() {
        Shutdown();
    }

    AsyncCompilationQueue(const AsyncCompilationQueue&) = delete;
    AsyncCompilationQueue& operator=(const AsyncCompilationQueue&) = delete;

    bool Submit(const Key& key, Job job) {
        std::scoped_lock lock{mutex};
        if (stopped || !in_flight.emplace(key).second) {
            return false;
        }
        jobs.emplace_back(key, std::move(job));
        work_ready.notify_one();
        return true;
    }

    std::optional<Completion> TryTake() {
        std::scoped_lock lock{mutex};
        if (completed.empty()) {
            return std::nullopt;
        }
        Completion completion = std::move(completed.front());
        completed.pop_front();
        in_flight.erase(completion.key);
        return completion;
    }

    std::optional<Completion> WaitTake() {
        std::unique_lock lock{mutex};
        completion_ready.wait(lock, [this] { return stopped || !completed.empty(); });
        if (completed.empty()) {
            return std::nullopt;
        }
        Completion completion = std::move(completed.front());
        completed.pop_front();
        in_flight.erase(completion.key);
        return completion;
    }

    void Shutdown() {
        {
            std::scoped_lock lock{mutex};
            if (stopped) {
                return;
            }
            stopped = true;
            jobs.clear();
        }
        work_ready.notify_all();
        completion_ready.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

private:
    void WorkerLoop() {
        for (;;) {
            std::pair<Key, Job> work;
            {
                std::unique_lock lock{mutex};
                work_ready.wait(lock, [this] { return stopped || !jobs.empty(); });
                if (stopped) {
                    return;
                }
                work = std::move(jobs.front());
                jobs.pop_front();
            }

            std::optional<Result> result;
            try {
                result.emplace(compiler(std::move(work.second)));
            } catch (...) {
                // A failed driver/compiler job is reported to the owner and must not terminate the
                // worker or publish a partial renderer object.
            }

            {
                std::scoped_lock lock{mutex};
                completed.push_back({std::move(work.first), std::move(result)});
            }
            completion_ready.notify_one();
        }
    }

    Compiler compiler;
    std::mutex mutex;
    std::condition_variable work_ready;
    std::condition_variable completion_ready;
    std::deque<std::pair<Key, Job>> jobs;
    std::deque<Completion> completed;
    std::unordered_set<Key, Hash> in_flight;
    bool stopped{};
    std::thread worker;
};

} // namespace Vulkan
