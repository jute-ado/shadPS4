// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <utility>

namespace AmdGpu {

template <typename IsSatisfied, typename PublishGpuWrites>
bool PollGpuMemoryWait(IsSatisfied&& is_satisfied, PublishGpuWrites&& publish_gpu_writes) {
    if (std::invoke(is_satisfied)) {
        return true;
    }
    std::invoke(std::forward<PublishGpuWrites>(publish_gpu_writes));
    return std::invoke(is_satisfied);
}

template <typename FinishGpu, typename ReadMemory>
void PublishGpuMemoryWait(bool has_pending_fence, bool& forced_completion, FinishGpu&& finish_gpu,
                          ReadMemory&& read_memory) {
    if (has_pending_fence && !forced_completion) {
        forced_completion = true;
        std::invoke(std::forward<FinishGpu>(finish_gpu));
        return;
    }
    std::invoke(std::forward<ReadMemory>(read_memory));
}

} // namespace AmdGpu
