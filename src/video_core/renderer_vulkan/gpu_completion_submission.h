// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <utility>

namespace Vulkan {

// Use the same lock as ordinary submissions. Enqueue only after a successful
// submission, using its returned signal tick, not the already-advanced CPU tick.
class GpuCompletionSubmission {
public:
    explicit GpuCompletionSubmission(std::mutex& submission_mutex)
        : submission_mutex{submission_mutex} {}

    template <typename Completion, typename SubmitGpuWork, typename DeferCompletion>
    [[nodiscard]] auto Submit(Completion&& completion, SubmitGpuWork&& submit_gpu_work,
                              DeferCompletion&& defer_completion) {
        std::scoped_lock lock{submission_mutex};
        const auto tick = std::forward<SubmitGpuWork>(submit_gpu_work)();
        std::forward<DeferCompletion>(defer_completion)(std::forward<Completion>(completion), tick);
        return tick;
    }

private:
    std::mutex& submission_mutex;
};

} // namespace Vulkan
