// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

namespace VideoCore {

class RenderDocCaptureState {
    enum class State {
        Idle,
        Triggered,
        Capturing,
    };

public:
    [[nodiscard]] bool Trigger() {
        auto expected = State::Idle;
        return state.compare_exchange_strong(expected, State::Triggered);
    }

    [[nodiscard]] bool TryBeginGuestSubmission(const bool has_guest_work) {
        if (!has_guest_work) {
            return false;
        }
        auto expected = State::Triggered;
        return state.compare_exchange_strong(expected, State::Capturing);
    }

    [[nodiscard]] bool FinishGuestSubmission() {
        auto expected = State::Capturing;
        return state.compare_exchange_strong(expected, State::Idle);
    }

private:
    std::atomic<State> state{State::Idle};
};

} // namespace VideoCore
