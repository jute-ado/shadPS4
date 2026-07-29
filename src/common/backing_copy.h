// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>

#include "common/types.h"

namespace Common {

inline bool CopyToWritableSpans(std::span<const u8> source,
                                std::span<const std::span<u8>> destinations) {
    size_t capacity = 0;
    for (const auto destination : destinations) {
        if (destination.size() > std::numeric_limits<size_t>::max() - capacity) {
            return false;
        }
        capacity += destination.size();
    }
    if (capacity < source.size()) {
        return false;
    }

    size_t source_offset = 0;
    for (const auto destination : destinations) {
        const size_t copy_size = std::min(destination.size(), source.size() - source_offset);
        if (copy_size == 0) {
            break;
        }
        std::memcpy(destination.data(), source.data() + source_offset, copy_size);
        source_offset += copy_size;
    }
    return source_offset == source.size();
}

} // namespace Common
