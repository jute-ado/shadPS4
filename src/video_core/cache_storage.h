// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/path_util.h"
#include "common/singleton.h"
#include "common/types.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include <queue>

namespace Storage {

namespace Detail {

// Serializes persistent-cache writes without making Save wait for disk I/O.
// Close establishes a barrier after the last accepted request and waits for
// that barrier before stopping the worker, so accepted writes cannot be lost.
class AsyncWriteQueue {
public:
    AsyncWriteQueue() = default;
    AsyncWriteQueue(const AsyncWriteQueue&) = delete;
    AsyncWriteQueue& operator=(const AsyncWriteQueue&) = delete;

    ~AsyncWriteQueue() {
        Close();
    }

    void Open(std::function<void()> worker_start = {}) {
        std::unique_lock lock{mutex};
        state_changed.wait(lock, [this] { return state != State::Closing; });
        if (state == State::Open) {
            return;
        }

        stop_requested = false;
        state = State::Open;
        try {
            worker = std::jthread{[this, worker_start = std::move(worker_start)] {
                if (worker_start) {
                    worker_start();
                }
                ProcessRequests();
            }};
        } catch (...) {
            state = State::Closed;
            state_changed.notify_all();
            throw;
        }
    }

    template <typename Function>
    [[nodiscard]] bool Submit(Function&& function) {
        return SubmitRequest(std::packaged_task<void()>{std::forward<Function>(function)});
    }

    // Waits until every request accepted before this call has completed. New
    // requests may still be accepted while Flush is waiting.
    void Flush() {
        std::future<void> barrier;
        {
            std::unique_lock lock{mutex};
            state_changed.wait(lock, [this] { return state != State::Closing; });
            if (state != State::Open) {
                return;
            }

            std::packaged_task<void()> barrier_request{[] {}};
            barrier = barrier_request.get_future();
            requests.emplace(std::move(barrier_request));
        }
        request_available.notify_one();
        barrier.wait();
    }

    // Atomically rejects new requests, queues a barrier after all accepted
    // requests, then stops the worker only after the barrier completes.
    void Close() {
        std::future<void> barrier;
        {
            std::unique_lock lock{mutex};
            if (state == State::Closing) {
                // This close is linearized with the close already in progress.
                // A concurrent Open may advance Closed to Open before this
                // waiter reacquires the lock, so only wait for Closing to end.
                state_changed.wait(lock, [this] { return state != State::Closing; });
                return;
            }
            if (state == State::Closed) {
                return;
            }

            state = State::Closing;
            std::packaged_task<void()> barrier_request{[] {}};
            barrier = barrier_request.get_future();
            requests.emplace(std::move(barrier_request));
        }
        request_available.notify_one();
        barrier.wait();

        {
            std::scoped_lock lock{mutex};
            stop_requested = true;
        }
        request_available.notify_one();
        worker.join();

        {
            std::scoped_lock lock{mutex};
            stop_requested = false;
            state = State::Closed;
        }
        state_changed.notify_all();
    }

private:
    enum class State {
        Closed,
        Open,
        Closing,
    };

    [[nodiscard]] bool SubmitRequest(std::packaged_task<void()>&& request) {
        {
            std::scoped_lock lock{mutex};
            if (state != State::Open) {
                return false;
            }
            requests.emplace(std::move(request));
        }
        request_available.notify_one();
        return true;
    }

    void ProcessRequests() {
        for (;;) {
            std::packaged_task<void()> request;
            {
                std::unique_lock lock{mutex};
                request_available.wait(lock,
                                       [this] { return stop_requested || !requests.empty(); });
                if (requests.empty()) {
                    if (stop_requested) {
                        return;
                    }
                    continue;
                }
                request = std::move(requests.front());
                requests.pop();
            }

            if (request.valid()) {
                request();
            }
        }
    }

    mutable std::mutex mutex;
    std::condition_variable request_available;
    std::condition_variable state_changed;
    std::queue<std::packaged_task<void()>> requests;
    std::jthread worker;
    State state{State::Closed};
    bool stop_requested{};
};

} // namespace Detail

enum class BlobType : u32 {
    ShaderMeta,
    ShaderBinary,
    PipelineKey,
    ShaderProfile,
};

class DataBase {
public:
    static DataBase& Instance() {
        return *Common::Singleton<DataBase>::Instance();
    }

    void Open();
    void Close();
    [[nodiscard]] bool IsOpened() const {
        return opened.load(std::memory_order_acquire);
    }
    void FinishPreload();

    bool Save(BlobType type, const std::string& name, std::vector<u8>&& data);
    bool Save(BlobType type, const std::string& name, std::vector<u32>&& data);

    void Load(BlobType type, const std::string& name, std::vector<u8>& data);
    void Load(BlobType type, const std::string& name, std::vector<u32>& data);

    void ForEachBlob(BlobType type, const std::function<void(std::vector<u8>&& data)>& func);

private:
    std::mutex lifecycle_mutex;
    Detail::AsyncWriteQueue io_queue;
    std::filesystem::path cache_path{};
    std::atomic_bool opened{};
};

} // namespace Storage
