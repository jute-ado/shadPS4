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

// Preserve the current loader behavior until the executable-placement
// invariant is specified by a failing test.
constexpr VAddr PreferredModuleLoadAddress(ModuleMappingRole, VAddr) noexcept {
    return LibraryModuleLoadBase;
}

} // namespace Core
