// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <optional>
#include <span>

#include "core/loader/elf.h"

namespace Core::Loader {

constexpr bool IsDynamicTableSizeAligned(std::size_t byte_size) {
    return byte_size % sizeof(elf_dynamic) == 0;
}

constexpr std::optional<std::size_t> FindDynamicTerminator(std::span<const elf_dynamic> entries) {
    for (std::size_t index = 0; index < entries.size(); ++index) {
        if (entries[index].d_tag == DT_NULL) {
            return index;
        }
    }
    return std::nullopt;
}

} // namespace Core::Loader
