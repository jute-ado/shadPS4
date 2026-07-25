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

} // namespace Core::Loader
