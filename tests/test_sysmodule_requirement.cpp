// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/sysmodule/sysmodule_requirement.h"

namespace Libraries::SysModule {
namespace {

TEST(SysModuleRequirement, ClassifiesLibcNeedMarker) {
    EXPECT_EQ(ClassifyPreloadRequirement("libc", "Need_sceLibc"),
              PreloadRequirement::Libc);
}

TEST(SysModuleRequirement, ClassifiesFiosInitializationMarkerIndependently) {
    EXPECT_EQ(ClassifyPreloadRequirement("libSceFios2", "sceFiosInitialize"),
              PreloadRequirement::Fios2);
}

TEST(SysModuleRequirement, RejectsUnrelatedLibrariesAndStubs) {
    EXPECT_EQ(ClassifyPreloadRequirement("libc", "malloc"), PreloadRequirement::None);
    EXPECT_EQ(ClassifyPreloadRequirement("libSceFios2", "sceFiosTerminate"),
              PreloadRequirement::None);
    EXPECT_EQ(ClassifyPreloadRequirement("libSceNet", "Need_sceLibc"),
              PreloadRequirement::None);
}

} // namespace
} // namespace Libraries::SysModule
