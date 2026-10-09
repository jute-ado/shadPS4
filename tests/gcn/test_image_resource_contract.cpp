// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <vector>

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

TEST(ShaderResourceContract, CachedExpandedSnapshotWithoutWalkerFailsClosed) {
    Info info{};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    info.user_data = user_data;
    info.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS + 4U;
    info.flattened_ud_buf.assign(info.srt_info.flattened_bufsize_dw, 0xDEADBEEFU);

    // Serialized metadata deliberately omits process-specific walker code.  An expanded flat
    // snapshot therefore cannot be refreshed from current user data and must not be replaced by
    // a zero-filled descriptor buffer that later specialization would treat as authoritative.
    EXPECT_FALSE(info.RefreshFlatBuf());
    EXPECT_TRUE(info.flattened_ud_buf.empty());
}

TEST(ShaderResourceContract, CachedUserDataOnlySnapshotRemainsRefreshableWithoutWalker) {
    Info info{};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    user_data[0] = 0x12345678U;
    user_data[1] = 0x9abcdef0U;
    info.user_data = user_data;
    info.srt_info.flattened_bufsize_dw = NUM_USER_DATA_REGS;
    info.flattened_ud_buf.assign(NUM_USER_DATA_REGS, 0xDEADBEEFU);

    EXPECT_TRUE(info.RefreshFlatBuf());
    EXPECT_EQ(info.flattened_ud_buf, std::vector<u32>(user_data.begin(), user_data.end()));
}

} // namespace
} // namespace Shader
