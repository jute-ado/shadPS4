// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "common/types.h"

namespace AmdGpu {

struct SubmissionProgress {
    u64 gfx_submissions;
    u64 interrupting_eops;
    u64 completed_interrupting_eops;
};

class SubmissionProgressTracker {
public:
    void ObserveGfxSubmission() noexcept {
        ++gfx_submissions;
    }

    void ObserveInterruptingEop() noexcept {
        ++interrupting_eops;
    }

    void ObserveInterruptingEopCompletion() noexcept {
        completed_interrupting_eops.fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] SubmissionProgress CompleteBoundary() noexcept {
        const u64 completed = TotalCompletedInterruptingEops();
        const SubmissionProgress progress{
            .gfx_submissions = gfx_submissions - prior_gfx_submissions,
            .interrupting_eops = interrupting_eops - prior_interrupting_eops,
            .completed_interrupting_eops = completed - prior_completed_interrupting_eops,
        };
        prior_gfx_submissions = gfx_submissions;
        prior_interrupting_eops = interrupting_eops;
        prior_completed_interrupting_eops = completed;
        return progress;
    }

    [[nodiscard]] u64 TotalCompletedInterruptingEops() const noexcept {
        return completed_interrupting_eops.load(std::memory_order_relaxed);
    }

private:
    u64 gfx_submissions{};
    u64 interrupting_eops{};
    std::atomic<u64> completed_interrupting_eops{};
    u64 prior_gfx_submissions{};
    u64 prior_interrupting_eops{};
    u64 prior_completed_interrupting_eops{};
};

} // namespace AmdGpu
