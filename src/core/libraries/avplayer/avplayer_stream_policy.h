// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>

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

} // namespace Libraries::AvPlayer
