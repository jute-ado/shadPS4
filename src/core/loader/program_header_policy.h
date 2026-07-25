// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

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

} // namespace Core::Loader
