// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>
#include <vector>

#include "common/unique_function.h"

namespace AmdGpu {

class EopCompletionBatch {
public:
    template <typename Completion>
    void Add(Completion&& completion) {
        completions.emplace_back(std::forward<Completion>(completion));
    }

    [[nodiscard]] bool Empty() const noexcept {
        return completions.empty();
    }

    template <typename DeferCompletion, typename SubmitGpuWork>
    bool Submit(DeferCompletion&& defer_completion, SubmitGpuWork&& submit_gpu_work) {
        if (completions.empty()) {
            return false;
        }

        auto pending = std::exchange(completions, {});
        std::forward<DeferCompletion>(defer_completion)(
            [pending = std::move(pending)]() mutable {
                for (auto& completion : pending) {
                    completion();
                }
            });
        std::forward<SubmitGpuWork>(submit_gpu_work)();
        return true;
    }

private:
    std::vector<Common::UniqueFunction<void>> completions;
};

} // namespace AmdGpu
