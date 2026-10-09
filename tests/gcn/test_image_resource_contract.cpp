// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <cstring>

#include "shader_recompiler/info.h"

namespace Shader {
namespace {

TEST(ShaderResourceContract, ReadUdSharpRejectsDescriptorPastSnapshot) {
    Info info{};
    info.flattened_ud_buf.assign(4U, 0xDEADBEEFU);

    // A descriptor read at dword 1 would need sixteen bytes.  The snapshot only has three
    // dwords remaining; the fail-closed result must not expose the trailing sentinel words.
    EXPECT_EQ(info.ReadUdSharp<u128>(1U), (u128{}));
}

TEST(ShaderResourceContract, ReadUdSharpCopiesACompleteDescriptorInBounds) {
    Info info{};
    info.flattened_ud_buf.resize(4U);
    const u128 expected{0x0123456789abcdefULL, 0xfedcba9876543210ULL};
    std::memcpy(info.flattened_ud_buf.data(), &expected, sizeof(expected));

    EXPECT_EQ(info.ReadUdSharp<u128>(0U), expected);
}

} // namespace
} // namespace Shader
