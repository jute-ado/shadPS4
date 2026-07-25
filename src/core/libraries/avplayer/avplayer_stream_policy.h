// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>

#include "common/types.h"

namespace Libraries::AvPlayer {

[[nodiscard]] constexpr bool HasDiscoveredStreams(std::size_t stream_count) {
    return stream_count != 0;
}

[[nodiscard]] constexpr bool IsValidSelectedStreamIndex(s32 index, std::size_t stream_count) {
    return index >= 0 && static_cast<std::size_t>(index) < stream_count;
}

[[nodiscard]] constexpr bool IsValidFfmpegStreamIndex(std::size_t index,
                                                       std::size_t stream_count) {
    return index < stream_count;
}

[[nodiscard]] constexpr std::optional<u32> ResolveFileReadRequest(u64 position, u64 file_size,
                                                                  s32 requested_size) {
    if (requested_size < 0) {
        return std::nullopt;
    }
    if (position >= file_size) {
        return 0;
    }
    return static_cast<u32>(
        std::min<u64>(static_cast<u32>(requested_size), file_size - position));
}

struct FileReadResult {
    s32 return_value;
    u64 next_position;
};

[[nodiscard]] constexpr FileReadResult ResolveFileReadResult(u64 position, u32 requested_size,
                                                              s32 callback_result) {
    if (callback_result <= 0) {
        return {.return_value = callback_result, .next_position = position};
    }
    const auto accepted_size =
        std::min(requested_size, static_cast<u32>(callback_result));
    const auto next_position =
        accepted_size > std::numeric_limits<u64>::max() - position
            ? std::numeric_limits<u64>::max()
            : position + accepted_size;
    return {.return_value = static_cast<s32>(accepted_size), .next_position = next_position};
}

[[nodiscard]] constexpr u64 ResolveFileSeek(u64 base_position, u64 file_size, s64 offset) {
    const auto base = std::min(base_position, file_size);
    if (offset >= 0) {
        const auto forward = static_cast<u64>(offset);
        return forward > file_size - base ? file_size : base + forward;
    }
    const auto backward = static_cast<u64>(-(offset + 1)) + 1;
    return backward > base ? 0 : base - backward;
}

} // namespace Libraries::AvPlayer
