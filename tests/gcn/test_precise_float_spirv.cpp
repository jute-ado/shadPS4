// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "common/object_pool.h"
#include "video_core/amdgpu/regs_shader.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/recompiler.h"
#include "spirv/1.0/GLSL.std.450.h"
#include "spirv/unified1/spirv.hpp"

namespace {

using namespace Shader;

std::vector<u32> EmitPreciseMultiplyAddShader(bool support_float64 = true,
                                              bool dx10_clamp = false) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.uses_lane_id = true;

    IR::Program program{info};
    Pools pools{};
    IR::Block* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    IR::IREmitter ir{*block};
    ir.Prologue();
    const auto value = ir.BitCast<IR::F32>(ir.LaneId());
    const auto product = ir.FPMul(value, ir.Imm32(1.5f));
    ir.Reference(ir.FPAdd(product, ir.Imm32(0.25f)));
    ir.Reference(ir.FPMul(value, ir.Imm32(2.0f)));
    ir.Reference(ir.FPSub(value, ir.Imm32(0.5f)));
    ir.Reference(ir.FPDiv(value, ir.Imm32(2.0f)));
    ir.Reference(ir.FPFma(value, ir.Imm32(2.0f), ir.Imm32(1.0f)));
    ir.Reference(ir.FPRecip(value));
    ir.Reference(ir.FPRecipSqrt(value));
    ir.Reference(ir.FPSqrt(value));
    ir.Reference(ir.FPAbs(value));
    ir.Reference(ir.FPNeg(value));
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.support_float64 = support_float64;
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(info.stage);
    runtime_info.dx10_clamp = dx10_clamp;
    runtime_info.cs_info.workgroup_size = {32, 1, 1};
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

u32 CountOpcode(const std::vector<u32>& spirv, spv::Op opcode) {
    u32 count{};
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        EXPECT_GT(words, 0u);
        count += static_cast<spv::Op>(word & 0xffff) == opcode;
        offset += words;
    }
    return count;
}

u32 CountDecoration(const std::vector<u32>& spirv, spv::Decoration decoration) {
    u32 count{};
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        EXPECT_GT(words, 0u);
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpDecorate && words >= 3 &&
            spirv[offset + 2] == static_cast<u32>(decoration)) {
            ++count;
        }
        offset += words;
    }
    return count;
}

u32 CountGlslExtInst(const std::vector<u32>& spirv, u32 instruction) {
    u32 count{};
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        EXPECT_GT(words, 0u);
        if (static_cast<spv::Op>(word & 0xffff) == spv::OpExtInst && words >= 5 &&
            spirv[offset + 4] == instruction) {
            ++count;
        }
        offset += words;
    }
    return count;
}

TEST(PreciseFloatSpirv, UsesExplicitFmaWithZeroToRoundMultiplyBeforeAdd) {
    const auto spirv = EmitPreciseMultiplyAddShader();

    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpFMul), 1);
    EXPECT_EQ(CountGlslExtInst(spirv, GLSLstd450Fma), 2);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpFAdd), 1);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpBitcast), 4);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpFConvert), 0);
    EXPECT_EQ(CountDecoration(spirv, spv::DecorationNoContraction), 0);
}

TEST(PreciseFloatSpirv, DoesNotRequireFloat64ForSeparateMultiplyRounding) {
    const auto spirv = EmitPreciseMultiplyAddShader(false);

    EXPECT_EQ(CountGlslExtInst(spirv, GLSLstd450Fma), 2);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpFConvert), 0);
    EXPECT_EQ(CountDecoration(spirv, spv::DecorationNoContraction), 0);
}

TEST(PreciseFloatSpirv, DecodesDx10ClampFromShaderProgramSettingsBit21) {
    AmdGpu::ShaderProgram program{};
    constexpr u32 dx10_clamp_bit = 1u << 21;
    std::memcpy(&program.settings, &dx10_clamp_bit, sizeof(dx10_clamp_bit));

    EXPECT_TRUE(program.settings.dx10_clamp);

    constexpr u32 adjacent_bits = (1u << 20) | (1u << 22) | (1u << 23);
    std::memcpy(&program.settings, &adjacent_bits, sizeof(adjacent_bits));
    EXPECT_FALSE(program.settings.dx10_clamp);
}

TEST(PreciseFloatSpirv, Dx10ClampDoesNotApplyToInternalIrOperations) {
    const auto spirv = EmitPreciseMultiplyAddShader(true, true);

    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpIsNan), 0);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpSelect), 0);
}

TEST(PreciseFloatSpirv, Dx10ClampParticipatesInShaderSpecializationIdentity) {
    RuntimeInfo disabled{};
    disabled.Initialize(Stage::Vertex);
    RuntimeInfo enabled = disabled;
    enabled.dx10_clamp = true;

    EXPECT_NE(disabled, enabled);
}

} // namespace
