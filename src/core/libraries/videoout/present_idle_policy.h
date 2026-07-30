// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Libraries::VideoOut {

enum class PresentIdleAction {
    None,
    BlankFrame,
    LastFrame,
};

[[nodiscard]] constexpr PresentIdleAction SelectPresentIdleAction(
    const bool has_pacing_budget, const bool video_port_open, const bool host_requires_redraw,
    const bool presented_screenshot_pending) {
    if (!has_pacing_budget) {
        return PresentIdleAction::None;
    }
    if (!video_port_open) {
        return PresentIdleAction::BlankFrame;
    }
    if (host_requires_redraw) {
        return PresentIdleAction::LastFrame;
    }
    return PresentIdleAction::None;
}

} // namespace Libraries::VideoOut
