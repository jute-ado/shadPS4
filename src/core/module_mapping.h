// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

#include "common/types.h"

namespace Core {

enum class ModuleMappingRole {
    MainExecutable,
    GameLibrary,
    SystemLibrary,
};

constexpr VAddr GameLibraryModuleLoadBase = 0x80000000;
constexpr VAddr SystemLibraryModuleLoadBase = 0x800000000;

constexpr VAddr PreferredModuleLoadAddress(ModuleMappingRole role,
                                           VAddr system_managed_base) noexcept {
    switch (role) {
    case ModuleMappingRole::MainExecutable:
        return system_managed_base;
    case ModuleMappingRole::GameLibrary:
        return GameLibraryModuleLoadBase;
    case ModuleMappingRole::SystemLibrary:
        return SystemLibraryModuleLoadBase;
    }
    std::unreachable();
}

} // namespace Core
