// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <unordered_map>

namespace Vulkan {

// Host shader and pipeline compilation does not exist on the guest GPU. Keep guest time and
// frame-lifecycle jobs stopped while any overlapping host compiler owns a lease, while preserving
// an existing user or debugger pause.
template <typename Controller>
class ScopedHostCompilationGuestPause final {
public:
    explicit ScopedHostCompilationGuestPause(Controller& controller_, bool enabled = true)
        : controller{controller_}, active{enabled} {
        if (!active) {
            return;
        }
        std::lock_guard lock{leases_mutex};
        auto& lease = leases[&controller];
        if (lease.owners++ == 0) {
            lease.resume_when_idle = !controller.IsGuestThreadsPaused();
            if (lease.resume_when_idle) {
                controller.PauseGuestThreads();
            }
        }
    }

    ~ScopedHostCompilationGuestPause() {
        if (!active) {
            return;
        }
        std::lock_guard lock{leases_mutex};
        const auto it = leases.find(&controller);
        if (it == leases.end()) {
            return;
        }
        if (--it->second.owners == 0) {
            // Resume before publishing an empty lease set. Otherwise a new compiler could observe
            // the old paused state between erase and resume and incorrectly decide that it does
            // not own the pause.
            if (it->second.resume_when_idle) {
                controller.ResumeGuestThreads();
            }
            leases.erase(it);
        }
    }

    ScopedHostCompilationGuestPause(const ScopedHostCompilationGuestPause&) = delete;
    ScopedHostCompilationGuestPause& operator=(const ScopedHostCompilationGuestPause&) = delete;

private:
    struct LeaseState {
        unsigned owners{};
        bool resume_when_idle{};
    };

    Controller& controller;
    bool active{};
    inline static std::mutex leases_mutex;
    inline static std::unordered_map<Controller*, LeaseState> leases;
};

} // namespace Vulkan
