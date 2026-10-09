// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <span>

#include <gtest/gtest.h>

#include "instructions.hpp"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/fetch_shader.h"

namespace Shader::Gcn {
namespace {

Shader::Info MakeInfo() {
    Shader::Info info{};
    info.has_fetch_shader = true;
    return info;
}

TEST(FetchShaderSpan, SliceReportsRemainingWordsBeforeDecode) {
    const std::array<u32, 2> words{0x12345678U, 0x9abcdef0U};
    GcnCodeSlice slice{words.data(), words.data() + words.size()};

    EXPECT_EQ(slice.remaining(), 2U);
    EXPECT_TRUE(slice.canReadWords(2));
    EXPECT_FALSE(slice.canReadWords(3));
    EXPECT_TRUE(slice.valid());
}

TEST(FetchShaderSpan, TruncatedWideInstructionIsRejected) {
    // The token selects a 64-bit encoding, but the supplied span has only one dword.
    const std::array<u32, 1> words{0xd0000000U};

    EXPECT_FALSE(ParseFetchShader(MakeInfo(), std::span<const u32>{words}).has_value());
}

TEST(FetchShaderSpan, CompleteTerminatorParsesFromCallerSpan) {
    const std::array<u32, 1> words{
        SOP1(OpcodeSOP1::S_SETPC_B64, SOperand7::S0, SOperand8::S0).Get(),
    };

    const auto parsed = ParseFetchShader(MakeInfo(), std::span<const u32>{words});
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->size, sizeof(u32));
}

} // namespace
} // namespace Shader::Gcn
