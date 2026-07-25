// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/loader/elf.h"

namespace Core::Loader {

enum class RelocationSymbolSource {
    Module,
    External,
    UndefinedWeak,
    Unsupported,
};

constexpr RelocationSymbolSource ClassifyRelocationSymbol(u8 binding, u64 value) {
    if (binding == STB_LOCAL || ((binding == STB_GLOBAL || binding == STB_WEAK) && value != 0)) {
        return RelocationSymbolSource::Module;
    }
    if (binding == STB_WEAK) {
        return RelocationSymbolSource::UndefinedWeak;
    }
    if (binding == STB_GLOBAL) {
        return RelocationSymbolSource::External;
    }
    return RelocationSymbolSource::Unsupported;
}

constexpr bool IsRelocationFunctionType(u8 type) {
    return type == STT_FUN || type == STT_SCE;
}

} // namespace Core::Loader
