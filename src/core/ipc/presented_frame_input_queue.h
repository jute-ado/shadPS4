// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <utility>

#include "common/types.h"

namespace Core::Ipc {

constexpr u32 MaxPresentedFrameInputEvents = 4096;
constexpr u64 MaxPresentedFrameInputFrame = 50'000'000;

[[nodiscard]] constexpr std::optional<u64> NextPresentedFrameForAutomation(
    const u64 current_presented_frame, const bool is_reusing_frame) noexcept {
    if (is_reusing_frame || current_presented_frame >= MaxPresentedFrameInputFrame) {
        return std::nullopt;
    }
    return current_presented_frame + 1;
}

enum class PresentedFrameInputKind : u8 {
    Button,
    Axis,
    ScreenshotWithOverlays,
};

struct PresentedFrameInputEvent {
    u64 presented_frame{};
    PresentedFrameInputKind kind{};
    u32 control{};
    u8 value{};
};

class PresentedFrameInputQueue {
public:
    [[nodiscard]] bool Schedule(const PresentedFrameInputEvent& event) noexcept {
        if (event.presented_frame == 0 || event.presented_frame > MaxPresentedFrameInputFrame) {
            return false;
        }
        if (size == events.size() && head != 0) {
            const u32 pending = size - head;
            for (u32 index = 0; index < pending; ++index) {
                events[index] = events[head + index];
            }
            head = 0;
            size = pending;
        }
        if (size == events.size()) {
            return false;
        }
        if (size != 0 && event.presented_frame < events[size - 1].presented_frame) {
            return false;
        }
        events[size++] = event;
        return true;
    }

    template <typename Callback>
    void Dispatch(const u64 presented_frame, Callback&& callback) {
        while (head < size && events[head].presented_frame <= presented_frame) {
            std::forward<Callback>(callback)(events[head++]);
        }
        if (head == size) {
            head = 0;
            size = 0;
        }
    }

    template <typename Callback>
    u32 DispatchExact(const u64 presented_frame, Callback&& callback) {
        u32 missed{};
        while (head < size && events[head].presented_frame < presented_frame) {
            ++head;
            ++missed;
        }
        while (head < size && events[head].presented_frame == presented_frame) {
            std::forward<Callback>(callback)(events[head++]);
        }
        if (head == size) {
            head = 0;
            size = 0;
        }
        return missed;
    }

    [[nodiscard]] u32 PendingCount() const noexcept {
        return size - head;
    }

private:
    std::array<PresentedFrameInputEvent, MaxPresentedFrameInputEvents> events{};
    u32 head{};
    u32 size{};
};

} // namespace Core::Ipc
