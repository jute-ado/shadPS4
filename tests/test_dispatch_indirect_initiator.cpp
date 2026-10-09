// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

namespace AmdGpu {
namespace {

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{input}, {}};
}

void ExpectPacketInitiatorCopiedBeforeDispatchGate(const std::string& source,
                                                   std::size_t begin,
                                                   std::size_t end,
                                                   const char* expected_field) {
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    ASSERT_LT(begin, end);

    const auto body = source.substr(begin, end - begin);
    const std::string assignment =
        std::string{"cs_program.dispatch_initiator = dispatch_indirect->"} + expected_field + ";";
    const auto assignment_pos = body.find(assignment);
    const auto gate_pos = body.find("if (rasterizer && (cs_program.dispatch_initiator & 1))");

    ASSERT_NE(assignment_pos, std::string::npos);
    ASSERT_NE(gate_pos, std::string::npos);
    EXPECT_LT(assignment_pos, gate_pos);
}

TEST(DispatchIndirectInitiator, GraphicsUsesPacketInitiatorForAdmission) {
    const auto source = ReadSource(SHADPS4_LIVERPOOL_SOURCE_PATH);
    ASSERT_FALSE(source.empty());

    const auto begin = source.find("case PM4ItOpcode::DispatchIndirect:");
    const auto end = source.find("case PM4ItOpcode::NumInstances:", begin);
    ExpectPacketInitiatorCopiedBeforeDispatchGate(source, begin, end, "dispatch_initiator");
}

TEST(DispatchIndirectInitiator, MecUsesPacketInitiatorForAdmission) {
    const auto source = ReadSource(SHADPS4_LIVERPOOL_SOURCE_PATH);
    ASSERT_FALSE(source.empty());

    const auto first = source.find("case PM4ItOpcode::DispatchIndirect:");
    const auto begin = source.find("case PM4ItOpcode::DispatchIndirect:", first + 1);
    const auto end = source.find("case PM4ItOpcode::WriteData:", begin);
    ExpectPacketInitiatorCopiedBeforeDispatchGate(source, begin, end, "dispatch_initiator");
}

} // namespace
} // namespace AmdGpu
