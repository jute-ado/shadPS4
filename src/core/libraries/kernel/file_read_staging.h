// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "common/types.h"

namespace Libraries::Kernel {

template <typename Reader>
size_t ReadFileThroughStaging(std::vector<u8>& staging, void* destination, size_t size,
                              Reader&& reader) {
    staging.resize(size);
    if (staging.empty()) {
        return 0;
    }

    const size_t bytes = std::forward<Reader>(reader)(std::span<u8>{staging});
    std::memcpy(destination, staging.data(), bytes);
    return bytes;
}

} // namespace Libraries::Kernel
