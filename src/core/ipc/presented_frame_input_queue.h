// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <deque>
#include <mutex>
#include <vector>

#include "common/types.h"

namespace Core::Ipc {

[[nodiscard]] constexpr bool AdvancesPresentedFrameAutomationClock(
    const bool is_reusing_frame) noexcept {
    return !is_reusing_frame;
}

enum class PresentedFrameInputType : u8 {
    Button,
    Axis,
    ScreenshotWithOverlays,
};

struct PresentedFrameInput {
    u64 frame{};
    PresentedFrameInputType type{};
    u32 code{};
    u8 value{};

    bool operator==(const PresentedFrameInput&) const = default;
};

class PresentedFrameInputQueue {
public:
    static constexpr std::size_t DefaultCapacity = 4096;

    explicit PresentedFrameInputQueue(const std::size_t capacity = DefaultCapacity)
        : capacity{capacity} {}

    [[nodiscard]] bool Push(const PresentedFrameInput& input) {
        std::scoped_lock lock{mutex};
        if (inputs.size() >= capacity || (has_last_frame && input.frame < last_frame)) {
            return false;
        }
        inputs.push_back(input);
        last_frame = input.frame;
        has_last_frame = true;
        return true;
    }

    [[nodiscard]] std::vector<PresentedFrameInput> PopDue(const u64 presented_frame) {
        std::scoped_lock lock{mutex};
        return PopDueLocked(presented_frame);
    }

    [[nodiscard]] std::vector<PresentedFrameInput> Advance() {
        std::scoped_lock lock{mutex};
        return PopDueLocked(++current_frame);
    }

    [[nodiscard]] std::size_t Size() const {
        std::scoped_lock lock{mutex};
        return inputs.size();
    }

private:
    [[nodiscard]] std::vector<PresentedFrameInput> PopDueLocked(const u64 presented_frame) {
        std::vector<PresentedFrameInput> due;
        while (!inputs.empty() && inputs.front().frame <= presented_frame) {
            due.push_back(inputs.front());
            inputs.pop_front();
        }
        return due;
    }
    const std::size_t capacity;
    mutable std::mutex mutex;
    std::deque<PresentedFrameInput> inputs;
    u64 last_frame{};
    u64 current_frame{};
    bool has_last_frame{};
};

} // namespace Core::Ipc
