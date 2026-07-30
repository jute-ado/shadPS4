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

    last_presented_frame = frame;
    free_queue.push(frame);
    return true;
}

} // namespace Vulkan
