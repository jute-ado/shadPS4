// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <limits>
#include <spirv/unified1/spirv.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv_quad_rect.h"
#include "shader_recompiler/runtime_info.h"

namespace {

size_t CountLocationDecorations(std::span<const u32> spirv) {
    size_t count{};
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 instruction = spirv[offset];
        const u16 word_count = instruction >> 16;
        const u16 opcode = instruction & 0xffff;
        if (opcode == static_cast<u16>(spv::Op::OpDecorate) && word_count >= 3 &&
            spirv[offset + 2] == static_cast<u32>(spv::DecorationLocation)) {
            ++count;
        }
        if (word_count == 0 || offset + word_count > spirv.size()) {
            return std::numeric_limits<size_t>::max();
        }
        offset += word_count;
    }
    return count;
}

} // namespace

TEST(AuxTessShader, FragmentlessPipelineDoesNotReuseAttributeLocations) {
    const auto spirv = Shader::Backend::SPIRV::EmitAuxilaryTessShader(
        Shader::Backend::SPIRV::AuxShaderType::RectListTCS, nullptr);

    EXPECT_EQ(CountLocationDecorations(spirv), 0);
}

TEST(AuxTessShader, FragmentInputsAreForwardedThroughControlShader) {
    Shader::FragmentRuntimeInfo fs_info{};
    fs_info.num_inputs = 1;
    fs_info.inputs[0] = {.param_index = 4, .is_default = false};

    const auto spirv = Shader::Backend::SPIRV::EmitAuxilaryTessShader(
        Shader::Backend::SPIRV::AuxShaderType::RectListTCS, &fs_info);

    EXPECT_EQ(CountLocationDecorations(spirv), 2);
}
