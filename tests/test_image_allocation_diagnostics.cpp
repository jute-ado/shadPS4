// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

namespace VideoCore {
namespace {

TEST(ImageAllocationDiagnostics, ReportsFailedRequestAndHeapBudgets) {
    std::ifstream source{SHADPS4_IMAGE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(source.is_open());
    const std::string text{std::istreambuf_iterator<char>{source},
                           std::istreambuf_iterator<char>{}};

    EXPECT_NE(text.find("vmaGetHeapBudgets"), std::string::npos);
    EXPECT_NE(text.find("image_ci.extent.width"), std::string::npos);
    EXPECT_NE(text.find("image_ci.extent.height"), std::string::npos);
    EXPECT_NE(text.find("image_ci.extent.depth"), std::string::npos);
    EXPECT_NE(text.find("image_ci.mipLevels"), std::string::npos);
    EXPECT_NE(text.find("image_ci.arrayLayers"), std::string::npos);
    EXPECT_NE(text.find("image_ci.samples"), std::string::npos);
    EXPECT_NE(text.find("image_ci.usage"), std::string::npos);
    EXPECT_NE(text.find("image_ci.flags"), std::string::npos);
}

} // namespace
} // namespace VideoCore
