// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace Libraries::AudioOut {

// SDL_GetAudioStreamQueued reports bytes in the stream's input format. The
// SDL backend feeds converted float frames, so the queue threshold must be
// based on that converted block rather than on the guest's original format.
constexpr std::uint32_t AudioQueueThresholdMultiplier = 4;

[[nodiscard]] constexpr std::uint32_t CalculateAudioQueueThreshold(
    const std::uint32_t converted_block_bytes, const std::uint32_t stream_frame_bytes,
    const int device_buffer_frames) noexcept {
    const std::uint64_t device_buffer_bytes =
        device_buffer_frames > 0
            ? static_cast<std::uint64_t>(device_buffer_frames) * stream_frame_bytes
            : 0;
    const std::uint64_t base_bytes =
        std::max<std::uint64_t>(converted_block_bytes, device_buffer_bytes);
    const std::uint64_t threshold = base_bytes * AudioQueueThresholdMultiplier;
    return threshold >= std::numeric_limits<std::uint32_t>::max()
               ? std::numeric_limits<std::uint32_t>::max()
               : static_cast<std::uint32_t>(threshold);
}

} // namespace Libraries::AudioOut
