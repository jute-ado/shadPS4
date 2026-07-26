// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <utility>

namespace Vulkan {

class GpuCompletionSubmission {
public:
    GpuCompletionSubmission() = default;

    explicit GpuCompletionSubmission(std::mutex& submission_mutex)
        : submission_mutex{&submission_mutex} {}

    GpuCompletionSubmission(const GpuCompletionSubmission&) = delete;
    GpuCompletionSubmission& operator=(const GpuCompletionSubmission&) = delete;

    template <typename Completion, typename GetSubmissionTick, typename DeferCompletion,
              typename SubmitGpuWork>
    void Submit(Completion&& completion, GetSubmissionTick&& get_submission_tick,
                DeferCompletion&& defer_completion, SubmitGpuWork&& submit_gpu_work) {
        std::scoped_lock lock{*submission_mutex};
        const auto tick = std::forward<GetSubmissionTick>(get_submission_tick)();
        std::forward<DeferCompletion>(defer_completion)(std::forward<Completion>(completion), tick);
        std::forward<SubmitGpuWork>(submit_gpu_work)();
    }

private:
    std::mutex owned_mutex;
    std::mutex* submission_mutex{&owned_mutex};
};

} // namespace Vulkan
