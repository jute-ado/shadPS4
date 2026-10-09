// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#include <sirit/sirit.h>

#include "shader_recompiler/backend/spirv/emit_spirv_quad_rect.h"
#include "shader_recompiler/runtime_info.h"

namespace {

size_t CountOpcode(std::span<const u32> module, spv::Op opcode) {
    constexpr size_t HeaderWords = 5;
    size_t count{};
    for (size_t offset = HeaderWords; offset < module.size();) {
        const u32 instruction = module[offset];
        const u32 word_count = instruction >> 16;
        EXPECT_NE(word_count, 0U);
        if (word_count == 0) {
            break;
        }
        if (static_cast<spv::Op>(instruction & 0xffffU) == opcode) {
            ++count;
        }
        offset += word_count;
    }
    return count;
}

TEST(AuxTessShader, PatchFactorsAreWrittenByOneInvocation) {
    const Shader::FragmentRuntimeInfo fragment_info{};
    for (const auto type : {Shader::Backend::SPIRV::AuxShaderType::RectListTCS,
                            Shader::Backend::SPIRV::AuxShaderType::QuadListTCS}) {
        const auto module = Shader::Backend::SPIRV::EmitAuxilaryTessShader(type, fragment_info);
        EXPECT_EQ(CountOpcode(module, spv::Op::OpSelectionMerge), 1U);
        EXPECT_EQ(CountOpcode(module, spv::Op::OpBranchConditional), 1U);
    }
}

} // namespace
