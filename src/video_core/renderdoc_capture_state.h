// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdint>

namespace VideoCore {

class RenderDocCaptureState {
    enum class State : std::uint32_t {
        Idle,
        Triggered,
        Capturing,
    };

    static constexpr std::uint32_t StateShift = 30;
    static constexpr std::uint32_t CountMask = (1U << StateShift) - 1;

    [[nodiscard]] static constexpr std::uint32_t Encode(const State state,
                                                        const std::uint32_t count) {
        return (static_cast<std::uint32_t>(state) << StateShift) | count;
    }

    [[nodiscard]] static constexpr State DecodeState(const std::uint32_t value) {
        return static_cast<State>(value >> StateShift);
    }

    [[nodiscard]] static constexpr std::uint32_t DecodeCount(const std::uint32_t value) {
        return value & CountMask;
    }

    [[nodiscard]] static constexpr std::uint32_t NormalizeCount(const std::uint32_t count) {
        return count == 0 ? 1 : count > CountMask ? CountMask : count;
    }

public:
    explicit RenderDocCaptureState(const std::uint32_t presented_frame_count = 1)
        : presented_frame_count{NormalizeCount(presented_frame_count)} {}

    [[nodiscard]] bool Trigger() {
        auto expected = Encode(State::Idle, 0);
        return state.compare_exchange_strong(expected,
                                             Encode(State::Triggered, presented_frame_count));
    }

    [[nodiscard]] bool ConsumePresentedFrameTrigger() {
        auto expected = state.load();
        while (DecodeState(expected) == State::Triggered) {
            const auto desired = Encode(State::Capturing, DecodeCount(expected));
            if (state.compare_exchange_weak(expected, desired)) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool FinishPresentedFrameCapture() {
        auto expected = state.load();
        while (DecodeState(expected) == State::Capturing) {
            const auto count = DecodeCount(expected);
            const auto desired = count == 1 ? Encode(State::Idle, 0)
                                            : Encode(State::Capturing, count - 1);
            if (state.compare_exchange_weak(expected, desired)) {
                return count == 1;
            }
        }
        return false;
    }

    [[nodiscard]] bool IsCapturing() const {
        return DecodeState(state.load()) == State::Capturing;
    }

private:
    const std::uint32_t presented_frame_count;
    std::atomic<std::uint32_t> state{Encode(State::Idle, 0)};
};

} // namespace VideoCore
