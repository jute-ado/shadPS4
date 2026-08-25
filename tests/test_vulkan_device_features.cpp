// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

namespace {

TEST(VulkanDeviceFeatures, EnablesSupportedIndirectFirstInstance) {
    std::ifstream source{SHADPS4_VK_INSTANCE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(source.is_open());
    const std::string text{std::istreambuf_iterator<char>{source},
                           std::istreambuf_iterator<char>{}};

    EXPECT_NE(text.find(".drawIndirectFirstInstance = features.drawIndirectFirstInstance"),
              std::string::npos);
}

} // namespace
