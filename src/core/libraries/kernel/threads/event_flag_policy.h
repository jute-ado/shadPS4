// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <utility>

namespace Libraries::Kernel {

enum class EventFlagQueueMode {
    Fifo,
    ThreadPriority,
};

enum class EventFlagWaitMode {
    And,
    Or,
};

enum class EventFlagClearMode {
    None,
    All,
    Bits,
};

[[nodiscard]] constexpr bool IsEventFlagWaitSatisfied(std::uint64_t current_bits,
                                                       std::uint64_t requested_bits,
                                                       EventFlagWaitMode mode) noexcept {
    return mode == EventFlagWaitMode::And ? (current_bits & requested_bits) == requested_bits
                                          : (current_bits & requested_bits) != 0;
}

[[nodiscard]] constexpr std::uint64_t ApplyEventFlagClear(std::uint64_t current_bits,
                                                          std::uint64_t requested_bits,
                                                          EventFlagClearMode mode) noexcept {
    switch (mode) {
    case EventFlagClearMode::None:
        return current_bits;
    case EventFlagClearMode::All:
        return 0;
    case EventFlagClearMode::Bits:
        return current_bits & ~requested_bits;
    }
    return current_bits;
}

template <typename WaitList, typename Waiter>
auto InsertEventFlagWaiter(WaitList& waiters, Waiter* waiter, EventFlagQueueMode mode) {
    if (mode == EventFlagQueueMode::Fifo) {
        waiters.push_back(waiter);
        return --waiters.end();
    }

    auto position = waiters.begin();
    while (position != waiters.end() && (*position)->priority <= waiter->priority) {
        ++position;
    }
    return waiters.insert(position, waiter);
}

template <typename WaitList, typename WakeWaiter>
void WakeEligibleEventFlagWaiters(WaitList& waiters, std::uint64_t& current_bits,
                                  WakeWaiter&& wake_waiter) {
    for (auto position = waiters.begin(); position != waiters.end();) {
        auto* waiter = *position;
        if (!IsEventFlagWaitSatisfied(current_bits, waiter->bits, waiter->wait_mode)) {
            ++position;
            continue;
        }
        waiter->result_bits = current_bits;
        current_bits = ApplyEventFlagClear(current_bits, waiter->bits, waiter->clear_mode);
        position = waiters.erase(position);
        std::forward<WakeWaiter>(wake_waiter)(waiter);
    }
}

} // namespace Libraries::Kernel
