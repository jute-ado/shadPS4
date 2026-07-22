// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <span>

#include "common/types.h"

namespace Core {

class BackingWriteCursor {
public:
    explicit BackingWriteCursor(std::span<const u8> source) : source{source} {}

    size_t Write(std::span<u8> destination) {
        const size_t copy_size = std::min(destination.size(), Remaining());
        std::memcpy(destination.data(), source.data() + offset, copy_size);
        offset += copy_size;
        return copy_size;
    }

    [[nodiscard]] size_t Remaining() const {
        return source.size() - offset;
    }

    [[nodiscard]] bool IsComplete() const {
        return Remaining() == 0;
    }

private:
    std::span<const u8> source;
    size_t offset{};
};

} // namespace Core
