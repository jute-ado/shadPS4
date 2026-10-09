// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

TEST(BufferThreadIdAddressing, UsesWaveLaneInEveryGraphicsAndComputeStage) {
    std::ifstream stream{SHADPS4_RESOURCE_TRACKING_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const std::string source{std::istreambuf_iterator<char>{stream},
                             std::istreambuf_iterator<char>{}};

    const auto branch = source.find("if (buffer.add_tid_enable)");
    ASSERT_NE(branch, std::string::npos);
    const auto branch_end = source.find("// offset =", branch);
    ASSERT_NE(branch_end, std::string::npos);
    const auto body = source.substr(branch, branch_end - branch);

    EXPECT_NE(body.find("ir.LaneId()"), std::string::npos);
    EXPECT_EQ(body.find("LogicalStage::Compute"), std::string::npos);
}
