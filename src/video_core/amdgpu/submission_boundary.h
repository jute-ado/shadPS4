// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

namespace AmdGpu {

template <typename Completion, typename SubmitGpuWork>
void SubmitSubmissionBoundary(Completion&& completion, SubmitGpuWork&& submit_gpu_work) {
    std::forward<SubmitGpuWork>(submit_gpu_work)();
    std::forward<Completion>(completion)();
}

} // namespace AmdGpu
