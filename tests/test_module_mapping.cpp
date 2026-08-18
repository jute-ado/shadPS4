// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/module_mapping.h"

TEST(ModuleMapping, MainExecutableStartsAtFirstUsableSystemManagedAddress) {
    constexpr VAddr system_managed_base = 0xc020000;
    EXPECT_EQ(Core::PreferredModuleLoadAddress(Core::ModuleMappingRole::MainExecutable,
                                              system_managed_base),
              system_managed_base);
}

TEST(ModuleMapping, GameLibrariesLoadBelowFourGiB) {
    EXPECT_EQ(Core::PreferredModuleLoadAddress(Core::ModuleMappingRole::GameLibrary, 0xc020000),
              Core::GameLibraryModuleLoadBase);
}

TEST(ModuleMapping, SystemLibrariesRetainTheHighModuleRegion) {
    EXPECT_EQ(Core::PreferredModuleLoadAddress(Core::ModuleMappingRole::SystemLibrary, 0xc020000),
              Core::SystemLibraryModuleLoadBase);
}
