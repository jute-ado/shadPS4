// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Vulkan {

// Host shader and pipeline compilation does not exist on the guest GPU. Keep guest time and
// frame-lifecycle jobs stopped while a driver performs that host-only work, while preserving an
// existing user or debugger pause.
template <typename Controller>
class ScopedHostCompilationGuestPause final {
public:
    explicit ScopedHostCompilationGuestPause(Controller& controller_) : controller{controller_} {
        owns_pause = !controller.IsGuestThreadsPaused();
        if (owns_pause) {
            controller.PauseGuestThreads();
        }
    }

    ~ScopedHostCompilationGuestPause() {
        if (owns_pause) {
            controller.ResumeGuestThreads();
        }
    }

    ScopedHostCompilationGuestPause(const ScopedHostCompilationGuestPause&) = delete;
    ScopedHostCompilationGuestPause& operator=(const ScopedHostCompilationGuestPause&) = delete;

private:
    Controller& controller;
    bool owns_pause{};
};

} // namespace Vulkan
