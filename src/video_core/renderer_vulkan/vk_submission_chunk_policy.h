// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>

#include "common/types.h"

namespace Vulkan {

class SubmissionChunkPolicy {
public:
    explicit constexpr SubmissionChunkPolicy(u32 work_budget) : work_budget{work_budget} {}

    [[nodiscard]] constexpr bool RecordGpuWork() noexcept {
        if (work_budget == 0) {
            return false;
        }
        if (++work_count < work_budget) {
            return false;
        }
        work_count = 0;
        return true;
    }

private:
    u32 work_budget{};
    u32 work_count{};
};

[[nodiscard]] constexpr std::optional<u64> SubmissionBackpressureWaitTick(
    u64 submitted_tick, u64 known_gpu_tick, u64 max_gpu_lead) noexcept {
    if (max_gpu_lead == 0 || submitted_tick <= known_gpu_tick + max_gpu_lead) {
        return std::nullopt;
    }
    return submitted_tick - max_gpu_lead;
}

} // namespace Vulkan
