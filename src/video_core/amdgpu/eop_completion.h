// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

namespace AmdGpu {

template <typename Packet, typename WriteMemory, typename SignalInterrupt,
          typename NotifyCompletion>
void PublishEop(Packet packet, WriteMemory&& write_memory, SignalInterrupt&& signal_interrupt,
                NotifyCompletion&& notify_completion) {
    packet.SignalFence(std::forward<WriteMemory>(write_memory),
                       std::forward<SignalInterrupt>(signal_interrupt));
    std::forward<NotifyCompletion>(notify_completion)();
}

} // namespace AmdGpu
