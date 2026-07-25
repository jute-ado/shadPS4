// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <limits>
#include <optional>
#include <span>

#include "core/loader/elf.h"

namespace Core::Loader {

enum class ProgramHeaderAction {
    Process,
    Ignore,
    Unsupported,
};

constexpr ProgramHeaderAction ClassifyProgramHeader(u32 type) {
    switch (type) {
    case PT_LOAD:
    case PT_DYNAMIC:
    case PT_TLS:
    case PT_SCE_DYNLIBDATA:
    case PT_SCE_PROCPARAM:
    case PT_SCE_RELRO:
    case PT_GNU_EH_FRAME:
        return ProgramHeaderAction::Process;
    case PT_NULL:
    case PT_INTERP:
    case PT_NOTE:
    case PT_PHDR:
    case PT_SCE_MODULE_PARAM:
    case PT_GNU_STACK:
    case PT_GNU_RELRO:
    case PT_SCE_COMMENT:
    case PT_SCE_LIBVERSION:
        return ProgramHeaderAction::Ignore;
    default:
        return ProgramHeaderAction::Unsupported;
    }
}

constexpr bool IsProgramHeaderFileSizeValid(const elf_program_header& header) {
    return (header.p_type != PT_LOAD && header.p_type != PT_SCE_RELRO) ||
           header.p_filesz <= header.p_memsz;
}

constexpr bool IsDirectlyLoadedFromFile(u32 type) {
    return type == PT_LOAD || type == PT_SCE_RELRO || type == PT_DYNAMIC ||
           type == PT_SCE_DYNLIBDATA;
}

constexpr std::optional<u64> GetAlignedSegmentSize(const elf_program_header& header) {
    const u64 alignment = header.p_align;
    if (alignment == 0 || alignment == 1) {
        return header.p_memsz;
    }
    if ((alignment & (alignment - 1)) != 0) {
        return std::nullopt;
    }
    const u64 padding = alignment - 1;
    if (header.p_memsz > std::numeric_limits<u64>::max() - padding) {
        return std::nullopt;
    }
    return (header.p_memsz + padding) & ~padding;
}

constexpr std::optional<u64> CalculateLoadImageSize(
    std::span<const elf_program_header> headers) {
    u64 image_size = 0;
    for (const auto& header : headers) {
        if (header.p_type != PT_LOAD && header.p_type != PT_SCE_RELRO) {
            continue;
        }
        const auto segment_size = GetAlignedSegmentSize(header);
        if (!segment_size ||
            header.p_vaddr > std::numeric_limits<u64>::max() - *segment_size) {
            return std::nullopt;
        }
        image_size = std::max(image_size, header.p_vaddr + *segment_size);
    }
    return image_size;
}

constexpr bool IsFileRangeValid(u64 file_size, u64 offset, u64 size) {
    return offset <= file_size && size <= file_size - offset;
}

constexpr bool IsRangeContained(u64 outer_offset, u64 outer_size, u64 inner_offset,
                                u64 inner_size) {
    if (outer_offset > std::numeric_limits<u64>::max() - outer_size ||
        inner_offset < outer_offset) {
        return false;
    }
    const u64 relative_offset = inner_offset - outer_offset;
    return relative_offset <= outer_size && inner_size <= outer_size - relative_offset;
}

constexpr std::optional<u64> ResolveSelfSegmentFileOffset(
    std::span<const self_segment_header> segments, std::span<const elf_program_header> headers,
    u64 file_size, u64 logical_offset, u64 size) {
    for (const auto& segment : segments) {
        if (!segment.IsBlocked()) {
            continue;
        }
        const u32 header_id = segment.GetId();
        if (header_id >= headers.size()) {
            continue;
        }
        const auto& header = headers[header_id];
        if (!IsRangeContained(header.p_offset, header.p_filesz, logical_offset, size) ||
            !IsFileRangeValid(file_size, segment.file_offset, segment.file_size)) {
            continue;
        }
        const u64 relative_offset = logical_offset - header.p_offset;
        if (segment.file_offset > std::numeric_limits<u64>::max() - relative_offset) {
            continue;
        }
        const u64 physical_offset = segment.file_offset + relative_offset;
        if (IsRangeContained(segment.file_offset, segment.file_size, physical_offset, size)) {
            return physical_offset;
        }
    }
    return std::nullopt;
}

constexpr std::optional<u64> ResolveFileTableOffset(u64 file_size, u64 base_offset,
                                                    u64 relative_offset, u64 entry_size,
                                                    u64 entry_count) {
    if (base_offset > std::numeric_limits<u64>::max() - relative_offset ||
        (entry_size != 0 && entry_count > std::numeric_limits<u64>::max() / entry_size)) {
        return std::nullopt;
    }
    const u64 table_offset = base_offset + relative_offset;
    const u64 table_size = entry_size * entry_count;
    if (!IsFileRangeValid(file_size, table_offset, table_size)) {
        return std::nullopt;
    }
    return table_offset;
}

constexpr std::optional<u64> ResolveElfHeaderTableOffset(
    u64 file_size, u64 elf_header_offset, u64 relative_offset, u64 declared_entry_size,
    u64 expected_entry_size, u64 entry_count) {
    if (entry_count != 0 && declared_entry_size != expected_entry_size) {
        return std::nullopt;
    }
    return ResolveFileTableOffset(file_size, elf_header_offset, relative_offset,
                                  expected_entry_size, entry_count);
}

constexpr std::optional<u64> ResolveSelfProgramIdOffset(u64 file_size, u64 declared_header_size,
                                                        u64 calculated_offset,
                                                        u64 program_id_size) {
    if (calculated_offset > declared_header_size ||
        program_id_size > declared_header_size - calculated_offset ||
        !IsFileRangeValid(file_size, calculated_offset, program_id_size)) {
        return std::nullopt;
    }
    return calculated_offset;
}

} // namespace Core::Loader
