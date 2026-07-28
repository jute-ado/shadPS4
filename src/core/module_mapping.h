// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Core {

enum class ModuleMappingRole {
    MainExecutable,
    Library,
};

constexpr VAddr LibraryModuleLoadBase = 0x800000000;

constexpr VAddr PreferredModuleLoadAddress(ModuleMappingRole role,
                                           VAddr system_managed_base) noexcept {
    return role == ModuleMappingRole::MainExecutable ? system_managed_base
                                                     : LibraryModuleLoadBase;
}

} // namespace Core
