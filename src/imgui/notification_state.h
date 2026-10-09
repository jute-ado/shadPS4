// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <optional>
#include <queue>
#include <utility>

namespace ImGui {

template <typename Notification>
class NotificationState {
public:
    void Push(Notification notification) {
        std::scoped_lock lock{pending_mutex};
        pending.push(std::move(notification));
    }

    bool ActivateNext() {
        if (current.has_value()) {
            return true;
        }

        std::scoped_lock lock{pending_mutex};
        if (pending.empty()) {
            return false;
        }
        current.emplace(std::move(pending.front()));
        pending.pop();
        return true;
    }

    bool CompleteCurrent() {
        current.reset();
        return ActivateNext();
    }

    Notification* Current() noexcept {
        return current ? &*current : nullptr;
    }

    const Notification* Current() const noexcept {
        return current ? &*current : nullptr;
    }

private:
    std::mutex pending_mutex;
    std::queue<Notification> pending;
    std::optional<Notification> current;
};

} // namespace ImGui
