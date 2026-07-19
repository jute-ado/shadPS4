// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "select_timing.h"

#include <chrono>
#include <thread>

namespace Libraries::Kernel {

bool WaitForSelectTimeout(s64 seconds, s64 microseconds) {
    if (seconds < 0 || microseconds < 0 || microseconds >= 1'000'000) {
        return false;
    }

    std::this_thread::sleep_for(std::chrono::seconds{seconds} +
                                std::chrono::microseconds{microseconds});
    return true;
}

} // namespace Libraries::Kernel
