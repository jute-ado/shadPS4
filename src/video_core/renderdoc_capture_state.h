// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdint>

namespace VideoCore {

class RenderDocCaptureState {
    enum class State {
        Idle,
        Triggered,
        Capturing,
    };

public:
    explicit RenderDocCaptureState(const std::uint32_t presented_frame_count = 1)
        : presented_frame_count{presented_frame_count} {}

    [[nodiscard]] bool Trigger() {
        auto expected = State::Idle;
        return state.compare_exchange_strong(expected, State::Triggered);
    }

    [[nodiscard]] bool ConsumePresentedFrameTrigger() {
        auto expected = State::Triggered;
        return state.compare_exchange_strong(expected, State::Capturing);
    }

    [[nodiscard]] bool FinishPresentedFrameCapture() {
        auto expected = State::Capturing;
        return state.compare_exchange_strong(expected, State::Idle);
    }

private:
    [[maybe_unused]] const std::uint32_t presented_frame_count;
    std::atomic<State> state{State::Idle};
};

} // namespace VideoCore
