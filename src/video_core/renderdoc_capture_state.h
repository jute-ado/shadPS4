// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

namespace VideoCore {

class RenderDocCaptureState {
    enum State : unsigned {
        Idle,
        Capturing,
        Triggered,
        // Larger values encode how many matching commands still need to be skipped.
        // This separate waiting state prevents an older pending submission from
        // claiming the request after the final skipped command, before a selected one.
        CountingMatches,
    };

public:
    [[nodiscard]] bool Trigger(unsigned match_skip = 0) {
        if (match_skip > 65535) {
            return false;
        }
        unsigned expected = State::Idle;
        const unsigned desired =
            match_skip == 0 ? State::Triggered : State::CountingMatches + match_skip;
        return state.compare_exchange_strong(expected, desired);
    }

    [[nodiscard]] bool SelectEligibleCommand(const bool matches) {
        if (!matches) {
            return false;
        }
        auto current = state.load();
        while (current >= State::Triggered) {
            if (current == State::Triggered) {
                return true;
            }
            const unsigned desired =
                current == State::CountingMatches ? State::Triggered : current - 1;
            if (state.compare_exchange_weak(current, desired)) {
                return desired == State::Triggered;
            }
        }
        return false;
    }

    [[nodiscard]] bool TryBeginGuestSubmission(const bool has_guest_work) {
        if (!has_guest_work) {
            return false;
        }
        unsigned expected = State::Triggered;
        return state.compare_exchange_strong(expected, State::Capturing);
    }

    [[nodiscard]] bool FinishGuestSubmission() {
        unsigned expected = State::Capturing;
        return state.compare_exchange_strong(expected, State::Idle);
    }

private:
    std::atomic<unsigned> state{State::Idle};
};

} // namespace VideoCore
