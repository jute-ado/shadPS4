// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

namespace AmdGpu {

template <typename Packet, typename DeferCompletion, typename SubmitGpuWork, typename WriteMemory,
          typename SignalInterrupt, typename NotifyCompletion>
void SubmitEop(Packet packet, DeferCompletion&& defer_completion,
               SubmitGpuWork&& submit_gpu_work, WriteMemory&& write_memory,
               SignalInterrupt&& signal_interrupt, NotifyCompletion&& notify_completion) {
    auto completion =
        [packet, write_memory = std::forward<WriteMemory>(write_memory),
         signal_interrupt = std::forward<SignalInterrupt>(signal_interrupt),
         notify_completion = std::forward<NotifyCompletion>(notify_completion)]() mutable {
            packet.SignalFence(write_memory, signal_interrupt);
            notify_completion();
        };
    std::forward<DeferCompletion>(defer_completion)(std::move(completion));
    std::forward<SubmitGpuWork>(submit_gpu_work)();
}

template <typename Packet, typename DeferCompletion, typename SubmitGpuWork, typename WriteMemory,
          typename SignalInterrupt>
void SubmitEop(Packet packet, DeferCompletion&& defer_completion,
               SubmitGpuWork&& submit_gpu_work, WriteMemory&& write_memory,
               SignalInterrupt&& signal_interrupt) {
    SubmitEop(std::move(packet), std::forward<DeferCompletion>(defer_completion),
              std::forward<SubmitGpuWork>(submit_gpu_work),
              std::forward<WriteMemory>(write_memory),
              std::forward<SignalInterrupt>(signal_interrupt), [] {});
}

} // namespace AmdGpu
