// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <span>
#include <vector>

#include <gtest/gtest.h>
#include <spirv/unified1/spirv.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv_discard_frag.h"
#include "shader_recompiler/runtime_info.h"

namespace {

struct SpirvInstruction {
    spv::Op opcode;
    std::span<const u32> operands;
};

std::vector<SpirvInstruction> DecodeInstructions(std::span<const u32> words) {
    std::vector<SpirvInstruction> instructions;
    if (words.size() < 5 || words.front() != spv::MagicNumber) {
        return instructions;
    }

    for (std::size_t index = 5; index < words.size();) {
        const u32 word_count = words[index] >> spv::WordCountShift;
        if (word_count == 0 || index + word_count > words.size()) {
            return {};
        }
        instructions.push_back({
            .opcode = static_cast<spv::Op>(words[index] & spv::OpCodeMask),
            .operands = words.subspan(index + 1, word_count - 1),
        });
        index += word_count;
    }
    return instructions;
}

std::size_t CountOpcode(std::span<const SpirvInstruction> instructions, spv::Op opcode) {
    return std::ranges::count(instructions, opcode, &SpirvInstruction::opcode);
}

} // namespace

TEST(ClipDistanceDiscardShader, EmitsOneDiscardTestPerExportedClipPlane) {
    std::array<Shader::OutputMap, 3> outputs{};
    outputs[0][1] = Shader::Output::ClipDist0;
    outputs[2][3] = Shader::Output::ClipDist7;

    const auto words = Shader::Backend::SPIRV::EmitDiscardFragmentShader(outputs);
    const auto instructions = DecodeInstructions(words);
    ASSERT_FALSE(instructions.empty());

    EXPECT_EQ(CountOpcode(instructions, spv::OpFOrdLessThan), 2);
    EXPECT_EQ(CountOpcode(instructions, spv::OpDemoteToHelperInvocationEXT), 2);

    std::vector<u32> locations;
    std::size_t fragment_entries = 0;
    std::size_t origin_upper_left_modes = 0;
    std::size_t input_variables = 0;
    for (const auto& instruction : instructions) {
        if (instruction.opcode == spv::OpEntryPoint && !instruction.operands.empty() &&
            instruction.operands[0] == spv::ExecutionModelFragment) {
            ++fragment_entries;
        } else if (instruction.opcode == spv::OpExecutionMode &&
                   instruction.operands.size() >= 2 &&
                   instruction.operands[1] == spv::ExecutionModeOriginUpperLeft) {
            ++origin_upper_left_modes;
        } else if (instruction.opcode == spv::OpVariable && instruction.operands.size() >= 3 &&
                   instruction.operands[2] == spv::StorageClassInput) {
            ++input_variables;
        } else if (instruction.opcode == spv::OpDecorate && instruction.operands.size() >= 3 &&
                   instruction.operands[1] == spv::DecorationLocation) {
            locations.push_back(instruction.operands[2]);
        }
    }

    std::ranges::sort(locations);
    EXPECT_EQ(fragment_entries, 1);
    EXPECT_EQ(origin_upper_left_modes, 1);
    EXPECT_EQ(input_variables, 2);
    EXPECT_EQ(locations, (std::vector<u32>{0, 2}));
}

TEST(ClipDistanceDiscardShader, IgnoresNonClipVertexOutputs) {
    std::array<Shader::OutputMap, 3> outputs{};
    outputs[0][0] = Shader::Output::PointSize;
    outputs[1][1] = Shader::Output::CullDist0;
    outputs[2][2] = Shader::Output::ViewportIndex;

    const auto words = Shader::Backend::SPIRV::EmitDiscardFragmentShader(outputs);
    const auto instructions = DecodeInstructions(words);
    ASSERT_FALSE(instructions.empty());

    EXPECT_EQ(CountOpcode(instructions, spv::OpFOrdLessThan), 0);
    EXPECT_EQ(CountOpcode(instructions, spv::OpDemoteToHelperInvocationEXT), 0);
}
