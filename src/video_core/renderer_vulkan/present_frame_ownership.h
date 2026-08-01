// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <queue>

namespace Vulkan {

template <typename Frame>
[[nodiscard]] bool CompletePresentFrameOwnership(std::queue<Frame*>& free_queue,
                                                 Frame*& last_presented_frame, Frame* frame,
                                                 const bool is_reusing_frame,
                                                 const bool was_presented) {
    if (is_reusing_frame) {
        return false;
    }

    if (!was_presented) {
        free_queue.push(frame);
        return true;
    }

    bool released_frame = false;
    if (last_presented_frame != nullptr) {
        free_queue.push(last_presented_frame);
        released_frame = true;
    }
    last_presented_frame = frame;
    return released_frame;
}

} // namespace Vulkan
