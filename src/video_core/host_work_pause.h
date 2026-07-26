// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <utility>

namespace VideoCore {

template <typename PauseGuestThreads, typename ResumeGuestThreads, typename HostWork>
void RunWithGuestThreadsPaused(PauseGuestThreads&& pause_guest_threads,
                               ResumeGuestThreads&& resume_guest_threads, HostWork&& host_work) {
    const bool owns_pause = std::invoke(pause_guest_threads);
    try {
        std::invoke(std::forward<HostWork>(host_work));
    } catch (...) {
        if (owns_pause) {
            std::invoke(resume_guest_threads);
        }
        throw;
    }
    if (owns_pause) {
        std::invoke(resume_guest_threads);
    }
}

} // namespace VideoCore
