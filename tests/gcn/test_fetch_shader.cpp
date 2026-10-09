// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <gtest/gtest.h>

#include "shader_recompiler/frontend/fetch_shader.h"

namespace {

Shader::Info MakeInfo(std::span<const u32> code) {
    Shader::Info info{};
    info.has_fetch_shader = true;
    info.fetch_shader_sgpr_base = 0;
    info.user_data = code;
    return info;
}

TEST(FetchShaderMalformed, TruncatedInstructionIsRejected) {
    // The first dword identifies a 64-bit instruction, but the backing span
    // contains only one dword. Parsing must fail closed rather than read past it.
    const std::array<u32, 1> truncated{0xD0000000u};
    EXPECT_FALSE(Shader::Gcn::ParseFetchShader(MakeInfo(truncated)).has_value());
}

TEST(FetchShaderMalformed, InvalidScalarDestinationIsRejected) {
    // S_LOAD_DWORD with sdst=104 (outside the 104-entry load table).
    const std::array<u32, 2> malformed{0x00520000u | (104u << 15), 0u};
    EXPECT_FALSE(Shader::Gcn::ParseFetchShader(MakeInfo(malformed)).has_value());
}

TEST(FetchShaderMalformed, InvalidSharpIndexIsRejected) {
    // A MUBUF instruction with src[2] selecting sharp index 104. The parser
    // must not index loads[104].
    const std::array<u32, 2> malformed{0xE0000000u, 0x80000000u | (26u << 43)};
    EXPECT_FALSE(Shader::Gcn::ParseFetchShader(MakeInfo(malformed)).has_value());
}

} // namespace
