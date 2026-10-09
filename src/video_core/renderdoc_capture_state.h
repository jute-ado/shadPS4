// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

namespace VideoCore {

class RenderDocCaptureState {
    enum class State {
        Idle,
        AwaitingFirstPresent,
        AwaitingSecondPresent,
    };

public:
    [[nodiscard]] bool TryBeginCapture() {
        auto expected = State::Idle;
        return state.compare_exchange_strong(expected, State::AwaitingFirstPresent);
    }

    [[nodiscard]] bool ConsumeCaptureForPresentedFrame(const bool is_reusing_frame) {
        if (is_reusing_frame) {
            return false;
        }
        auto expected = State::AwaitingFirstPresent;
        if (state.compare_exchange_strong(expected, State::AwaitingSecondPresent)) {
            return false;
        }
        expected = State::AwaitingSecondPresent;
        return state.compare_exchange_strong(expected, State::Idle);
    }

private:
    std::atomic<State> state{State::Idle};
};

} // namespace VideoCore
