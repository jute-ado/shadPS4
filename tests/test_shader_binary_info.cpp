// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <gtest/gtest.h>

#include "video_core/amdgpu/regs_shader.h"

namespace {

TEST(ShaderBinaryInfo, MissingMetadataIsRejectedWithoutUnreachable) {
    // SearchBinaryInfo scans a bounded 0x4000-dword window. Keep the backing storage at
    // least that large so the malformed input is deterministic and does not rely on a guard page.
    std::array<u32, 0x4000> code{};
    const auto info = AmdGpu::SearchBinaryInfo(code.data());
    EXPECT_FALSE(info.has_value());
}

TEST(ShaderBinaryInfo, ValidMetadataStillProducesShaderParameters) {
    std::array<u32, 0x4000> code{};
    const auto info_offset = 32U;
    auto* info = reinterpret_cast<AmdGpu::BinaryInfo*>(code.data() + info_offset);
    info->signature = AmdGpu::BinaryInfo::signature_ref;
    info->length = 8;
    info->shader_hash = 0x123456789abcdef0ULL;

    const auto found = AmdGpu::SearchBinaryInfo(code.data());
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(*found, info);
}

TEST(ShaderBinaryInfo, ZeroLengthMetadataIsRejected) {
    std::array<u32, 0x4000> code{};
    auto* info = reinterpret_cast<AmdGpu::BinaryInfo*>(code.data() + 32);
    info->signature = AmdGpu::BinaryInfo::signature_ref;
    info->length = 0;

    EXPECT_FALSE(AmdGpu::SearchBinaryInfo(code.data()).has_value());
}

} // namespace
