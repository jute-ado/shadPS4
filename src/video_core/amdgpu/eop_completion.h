// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

#include "common/unique_function.h"

namespace AmdGpu {

template <typename Packet, typename WriteMemory, typename SignalInterrupt,
          typename NotifyCompletion>
void PublishEop(Packet packet, WriteMemory&& write_memory, SignalInterrupt&& signal_interrupt,
                NotifyCompletion&& notify_completion) {
    packet.SignalFence(std::forward<WriteMemory>(write_memory),
                       std::forward<SignalInterrupt>(signal_interrupt));
    std::forward<NotifyCompletion>(notify_completion)();
}

template <typename Packet, typename SubmitWithCompletion, typename WriteMemory,
          typename SignalInterrupt, typename NotifyCompletion>
void SubmitEopAtGpuCompletion(Packet packet, SubmitWithCompletion&& submit_with_completion,
                              WriteMemory&& write_memory, SignalInterrupt&& signal_interrupt,
                              NotifyCompletion&& notify_completion) {
    auto completion =
        [packet = std::move(packet), write_memory = std::forward<WriteMemory>(write_memory),
         signal_interrupt = std::forward<SignalInterrupt>(signal_interrupt),
         notify_completion = std::forward<NotifyCompletion>(notify_completion)]() mutable {
            PublishEop(std::move(packet), write_memory, signal_interrupt, notify_completion);
        };
    std::forward<SubmitWithCompletion>(submit_with_completion)(
        Common::UniqueFunction<void>{std::move(completion)});
}

} // namespace AmdGpu
