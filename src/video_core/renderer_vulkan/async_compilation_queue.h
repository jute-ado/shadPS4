// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
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
    using CompletionNotifier = std::function<void()>;

    explicit AsyncCompilationQueue(
        Compiler compiler_, size_t max_outstanding_ = std::numeric_limits<size_t>::max(),
        CompletionNotifier completion_notifier_ = {})
        : compiler{std::move(compiler_)}, max_outstanding{max_outstanding_},
          completion_notifier{std::move(completion_notifier_)},
          worker{[this] { WorkerLoop(); }} {}

    ~AsyncCompilationQueue() {
        Shutdown();
    }

    AsyncCompilationQueue(const AsyncCompilationQueue&) = delete;
    AsyncCompilationQueue& operator=(const AsyncCompilationQueue&) = delete;

    bool Submit(const Key& key, Job job) {
        std::scoped_lock lock{mutex};
        if (stopped || in_flight.contains(key) || in_flight.size() >= max_outstanding) {
            return false;
        }
        in_flight.emplace(key);
        jobs.emplace_back(key, std::move(job));
        work_ready.notify_one();
        return true;
    }

    [[nodiscard]] bool CanAccept() {
        std::scoped_lock lock{mutex};
        return !stopped && in_flight.size() < max_outstanding;
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
            if (completion_notifier) {
                completion_notifier();
            }
        }
    }

    Compiler compiler;
    size_t max_outstanding;
    CompletionNotifier completion_notifier;
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
