// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace Shader {

struct SrtWalkerCodeRange {
    std::uintptr_t begin{std::numeric_limits<std::uintptr_t>::max()};
    std::uintptr_t end{};

    constexpr void Include(std::uintptr_t address, std::size_t size) noexcept {
        if (size == 0) {
            return;
        }
        const auto maximum = std::numeric_limits<std::uintptr_t>::max();
        const auto range_end = size > maximum - address ? maximum : address + size;
        begin = std::min(begin, address);
        end = std::max(end, range_end);
    }

    [[nodiscard]] constexpr bool Contains(std::uintptr_t address) const noexcept {
        return begin <= address && address < end;
    }
};

} // namespace Shader
