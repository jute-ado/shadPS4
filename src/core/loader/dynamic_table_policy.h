// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

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

inline std::optional<std::span<const char>> FindDynamicStringTable(
    std::span<const elf_dynamic> entries, std::span<const u8> dynamic_data) {
    std::optional<u64> offset;
    std::optional<u64> size;
    for (const auto& entry : entries) {
        if (entry.d_tag == DT_SCE_STRTAB) {
            offset = entry.d_un.d_ptr;
        } else if (entry.d_tag == DT_SCE_STRSZ) {
            size = entry.d_un.d_val;
        }
    }
    if (!offset || !size || *offset > dynamic_data.size() ||
        *size > dynamic_data.size() - *offset) {
        return std::nullopt;
    }
    return std::span{reinterpret_cast<const char*>(dynamic_data.data() + *offset),
                     static_cast<std::size_t>(*size)};
}

constexpr std::optional<std::string_view> ReadDynamicString(std::span<const char> table,
                                                            u64 offset) {
    if (offset >= table.size()) {
        return std::nullopt;
    }
    const auto start = static_cast<std::size_t>(offset);
    for (std::size_t end = start; end < table.size(); ++end) {
        if (table[end] == '\0') {
            return std::string_view{table.data() + start, end - start};
        }
    }
    return std::nullopt;
}

template <typename T>
inline std::optional<std::span<const T>> FindDynamicDataTable(
    std::span<const elf_dynamic> entries, std::span<const u8> dynamic_data, s64 offset_tag,
    s64 size_tag) {
    std::optional<u64> offset;
    std::optional<u64> size;
    for (const auto& entry : entries) {
        if (entry.d_tag == offset_tag) {
            offset = entry.d_un.d_ptr;
        } else if (entry.d_tag == size_tag) {
            size = entry.d_un.d_val;
        }
    }
    if (!offset || !size || *offset > dynamic_data.size() ||
        *size > dynamic_data.size() - *offset || *size % sizeof(T) != 0) {
        return std::nullopt;
    }
    const auto* bytes = dynamic_data.data() + *offset;
    if (reinterpret_cast<std::uintptr_t>(bytes) % alignof(T) != 0) {
        return std::nullopt;
    }
    return std::span{reinterpret_cast<const T*>(bytes),
                     static_cast<std::size_t>(*size / sizeof(T))};
}

} // namespace Core::Loader
