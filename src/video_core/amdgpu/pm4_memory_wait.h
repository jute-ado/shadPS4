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

} // namespace AmdGpu
