// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

namespace AmdGpu {

template <typename Packet, typename SubmitGpuWork, typename WriteMemory, typename SignalInterrupt>
void SubmitEop(Packet packet, SubmitGpuWork&& submit_gpu_work, WriteMemory&& write_memory,
               SignalInterrupt&& signal_interrupt) {
    std::forward<SubmitGpuWork>(submit_gpu_work)();
    packet.SignalFence(std::forward<WriteMemory>(write_memory),
                       std::forward<SignalInterrupt>(signal_interrupt));
}

} // namespace AmdGpu
