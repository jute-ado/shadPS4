// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

namespace VideoCore {

class RenderDocCaptureState {
    enum class State {
        Idle,
        Triggered,
        ArmedForGuestWork,
        Capturing,
    };

public:
    static constexpr unsigned GuestSubmissionLimit = 16;

    [[nodiscard]] bool Trigger() {
        auto expected = State::Idle;
        return state.compare_exchange_strong(expected, State::Triggered);
    }

    [[nodiscard]] bool ConsumeGuestFrameTrigger() {
        auto expected = State::ArmedForGuestWork;
        if (!state.compare_exchange_strong(expected, State::Capturing)) {
            return false;
        }
        guest_submissions.store(0);
        return true;
    }

    [[nodiscard]] bool ArmForPreparedGuestWork() {
        auto expected = State::Triggered;
        return state.compare_exchange_strong(expected, State::ArmedForGuestWork);
    }

    [[nodiscard]] bool IsCapturing() const {
        return state.load() == State::Capturing;
    }

    [[nodiscard]] bool FinishPresentedFrameCapture() {
        auto expected = State::Capturing;
        return state.compare_exchange_strong(expected, State::Idle);
    }

    [[nodiscard]] bool ShouldFinishAfterGuestSubmission() {
        if (!IsCapturing()) {
            return false;
        }
        return guest_submissions.fetch_add(1) + 1 >= GuestSubmissionLimit;
    }

private:
    std::atomic<State> state{State::Idle};
    std::atomic<unsigned> guest_submissions{0};
};

} // namespace VideoCore
