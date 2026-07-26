// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

namespace AmdGpu {

template <typename Completion, typename DeferCompletion, typename SubmitGpuWork>
void SubmitSubmissionBoundary(Completion&& completion, DeferCompletion&& defer_completion,
                              SubmitGpuWork&& submit_gpu_work) {
    std::forward<DeferCompletion>(defer_completion)(std::forward<Completion>(completion));
    std::forward<SubmitGpuWork>(submit_gpu_work)();
}

} // namespace AmdGpu
