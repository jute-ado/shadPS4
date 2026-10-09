// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

namespace AmdGpu {

class EopSubmissionBatch {
public:
    void MarkEopPending() {
        has_pending_eop = true;
    }

    template <typename PrepareGpuWork, typename SubmitGpuWork>
    void SubmitBoundary(PrepareGpuWork&& prepare_gpu_work, SubmitGpuWork&& submit_gpu_work) {
        has_pending_eop = false;
        std::forward<PrepareGpuWork>(prepare_gpu_work)();
        std::forward<SubmitGpuWork>(submit_gpu_work)();
    }

    template <typename PrepareGpuWork, typename SubmitGpuWork>
    void FlushIfPending(PrepareGpuWork&& prepare_gpu_work, SubmitGpuWork&& submit_gpu_work) {
        if (!has_pending_eop) {
            return;
        }
        has_pending_eop = false;
        std::forward<PrepareGpuWork>(prepare_gpu_work)();
        std::forward<SubmitGpuWork>(submit_gpu_work)();
    }

private:
    bool has_pending_eop{};
};

} // namespace AmdGpu
