// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

TEST(BufferThreadIdAddressing, UsesWaveLaneOnlyForComputeAndRejectsGraphics) {
    std::ifstream stream{SHADPS4_RESOURCE_TRACKING_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const std::string source{std::istreambuf_iterator<char>{stream},
                             std::istreambuf_iterator<char>{}};

    const auto branch = source.find("if (buffer.add_tid_enable)");
    ASSERT_NE(branch, std::string::npos);
    const auto branch_end = source.find("// offset =", branch);
    ASSERT_NE(branch_end, std::string::npos);
    const auto body = source.substr(branch, branch_end - branch);

    // ADD_TID is a guest wavefront (0..63) operation.  The compute path has
    // an explicit wave-64 lowering, while graphics may run in a host subgroup
    // half and must fail translation rather than silently use the wrong lane.
    const auto stage_guard = body.find("LogicalStage::Compute");
    ASSERT_NE(stage_guard, std::string::npos);
    EXPECT_NE(body.find("info.translation_failed", stage_guard), std::string::npos);
    EXPECT_NE(body.find("ir.LaneId()", stage_guard), std::string::npos);
}
