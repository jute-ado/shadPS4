// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include "core/libraries/sysmodule/sysmodule_requirement.h"

namespace Libraries::SysModule {
namespace {

std::string ReadSysmoduleInternalSource() {
    std::ifstream source{SHADPS4_SYSMODULE_INTERNAL_SOURCE_PATH, std::ios::binary};
    EXPECT_TRUE(source.is_open());
    std::string text{std::istreambuf_iterator<char>{source}, std::istreambuf_iterator<char>{}};
    std::erase(text, '\r');
    return text;
}

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

TEST(SysModuleLlePolicy, AllowsComputerVisionLibrariesWhenFirmwareModulesAreAvailable) {
    const std::string source = ReadSysmoduleInternalSource();

    EXPECT_NE(source.find("{\"libSceDepth.sprx\", nullptr}"), std::string::npos);
    EXPECT_NE(source.find("{\"libScePadTracker.sprx\", nullptr}"), std::string::npos);
    EXPECT_NE(source.find("{\"libSceMoveTracker.sprx\", nullptr}"), std::string::npos);
}

} // namespace
} // namespace Libraries::SysModule
