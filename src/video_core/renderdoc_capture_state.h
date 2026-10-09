// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

namespace VideoCore {

class RenderDocCaptureState {
    enum class State {
        Idle,
        Triggered,
    };

public:
    [[nodiscard]] bool Trigger() {
        auto expected = State::Idle;
        return state.compare_exchange_strong(expected, State::Triggered);
    }

    [[nodiscard]] bool ConsumePresentedFrameTrigger() {
        auto expected = State::Triggered;
        return state.compare_exchange_strong(expected, State::Idle);
    }

private:
    std::atomic<State> state{State::Idle};
};

} // namespace VideoCore
