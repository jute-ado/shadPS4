// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <limits>

#include "common/types.h"

namespace VideoCore {

struct BufferReadbackWindow {
    VAddr download_address{};
    u64 download_size{};
    VAddr request_address{};
    u64 request_size{};
    bool valid{};
};

[[nodiscard]] constexpr BufferReadbackWindow PlanBufferReadbackWindow(
    VAddr buffer_address, u64 buffer_size, VAddr request_address, u64 request_size,
    u64 window_size) noexcept {
    if (buffer_size == 0 || request_size == 0 || window_size == 0 ||
        (window_size & (window_size - 1)) != 0 ||
        buffer_size > std::numeric_limits<VAddr>::max() - buffer_address ||
        request_size > std::numeric_limits<VAddr>::max() - request_address) {
        return {};
    }

    const VAddr buffer_end = buffer_address + buffer_size;
    const VAddr request_end = request_address + request_size;
    if (request_address < buffer_address || request_end > buffer_end) {
        return {};
    }

    const VAddr aligned_request = request_address & ~(window_size - 1);
    const VAddr download_address = std::max(aligned_request, buffer_address);
    const VAddr window_end =
        window_size > std::numeric_limits<VAddr>::max() - download_address
            ? std::numeric_limits<VAddr>::max()
            : download_address + window_size;
    const VAddr download_end = std::min(std::max(window_end, request_end), buffer_end);
    if (download_end <= download_address) {
        return {};
    }

    return {
        .download_address = download_address,
        .download_size = download_end - download_address,
        .request_address = request_address,
        .request_size = request_size,
        .valid = true,
    };
}

} // namespace VideoCore
