// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include <gtest/gtest.h>

#include "common/object_pool.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/recompiler.h"
#include "spirv/unified1/GLSL.std.450.h"
#include "spirv/unified1/spirv.hpp"

namespace {

using namespace Shader;

std::vector<u32> EmitNonAmdBarycentricShader(IR::Attribute attribute) {
    Info info{};
    info.stage = Stage::Fragment;
    info.l_stage = LogicalStage::Fragment;
    info.loads.Set(attribute, 0);

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
    ir.Reference(ir.GetAttribute(attribute));
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.supports_fragment_shader_barycentric = true;
    profile.supports_amd_shader_explicit_vertex_parameter = false;

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Fragment);
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

size_t CountOpcode(const std::vector<u32>& spirv, spv::Op opcode) {
    size_t count = 0;
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == opcode) {
            ++count;
        }
        EXPECT_GT(words, 0u);
        offset += words;
    }
    return count;
}

size_t CountGlslExtInst(const std::vector<u32>& spirv, u32 instruction) {
    size_t count = 0;
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::OpExtInst && words >= 5 &&
            spirv[offset + 4] == instruction) {
            ++count;
        }
        offset += words;
    }
    return count;
}

bool HasCapability(const std::vector<u32>& spirv, spv::Capability capability) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpCapability && words >= 2 &&
            spirv[offset + 1] == static_cast<u32>(capability)) {
            return true;
        }
        offset += words;
    }
    return false;
}

bool HasBuiltIn(const std::vector<u32>& spirv, spv::BuiltIn builtin) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpDecorate && words >= 4 &&
            spirv[offset + 2] == static_cast<u32>(spv::DecorationBuiltIn) &&
            spirv[offset + 3] == static_cast<u32>(builtin)) {
            return true;
        }
        offset += words;
    }
    return false;
}

struct InterpMovIrSummary {
    std::vector<u32> per_vertex_indices;
    std::size_t subtraction_count{};
    Interpolation interpolation{};
};

InterpMovIrSummary TranslateInterpMov(const u32 parameter) {
    Info info{};
    info.stage = Stage::Fragment;
    info.l_stage = LogicalStage::Fragment;
    IR::Program program{info};
    Pools pools{};
    IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.supports_fragment_shader_barycentric = true;

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Fragment);
    runtime_info.fs_info.num_inputs = 1;
    runtime_info.fs_info.inputs[0] = {
        .param_index = 0,
        .is_default = false,
        .is_flat = true,
        .default_value = 0,
    };

    Gcn::Translator translator{program.info, runtime_info, profile};
    translator.EmitPrologue(block);
    Gcn::GcnInst instruction{};
    instruction.control.vintrp.attr = 0;
    instruction.control.vintrp.chan = 0;
    instruction.src[0].code = parameter;
    instruction.dst[0].field = Gcn::OperandField::VectorGPR;
    instruction.dst[0].code = 0;
    translator.V_INTERP_MOV_F32(instruction);

    InterpMovIrSummary summary{.interpolation = program.info.fs_interpolation[0]};
    for (const IR::Inst& inst : *block) {
        if (inst.GetOpcode() == IR::Opcode::GetAttribute) {
            summary.per_vertex_indices.push_back(inst.Arg(2).U32());
        } else if (inst.GetOpcode() == IR::Opcode::FPSub32) {
            ++summary.subtraction_count;
        }
    }
    std::ranges::sort(summary.per_vertex_indices);
    return summary;
}

TEST(BarycentricSpirv, NonAmdSmoothUsesKhrInputDirectly) {
    const auto spirv = EmitNonAmdBarycentricShader(IR::Attribute::BaryCoordSmooth);

    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityFragmentBarycentricKHR));
    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityInterpolationFunction));
    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInBaryCoordKHR));
    EXPECT_EQ(CountGlslExtInst(spirv, GLSLstd450InterpolateAtCentroid), 0u);
    EXPECT_EQ(CountGlslExtInst(spirv, GLSLstd450InterpolateAtSample), 0u);
}

TEST(BarycentricSpirv, NonAmdCentroidUsesInterpolationFunction) {
    const auto spirv = EmitNonAmdBarycentricShader(IR::Attribute::BaryCoordSmoothCentroid);

    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityInterpolationFunction));
    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInBaryCoordKHR));
    EXPECT_EQ(CountGlslExtInst(spirv, GLSLstd450InterpolateAtCentroid), 1u);
}

TEST(BarycentricSpirv, NonAmdSampleUsesSampleIdAndSampleRateShading) {
    const auto spirv = EmitNonAmdBarycentricShader(IR::Attribute::BaryCoordSmoothSample);

    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityInterpolationFunction));
    EXPECT_TRUE(HasCapability(spirv, spv::CapabilitySampleRateShading));
    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInBaryCoordKHR));
    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInSampleId));
    EXPECT_EQ(CountGlslExtInst(spirv, GLSLstd450InterpolateAtSample), 1u);
}

TEST(VectorInterpolation, InterpMovP10UsesFirstVertexDelta) {
    const auto summary = TranslateInterpMov(0);

    EXPECT_EQ(summary.per_vertex_indices, (std::vector<u32>{0, 1}));
    EXPECT_EQ(summary.subtraction_count, 1u);
    EXPECT_EQ(summary.interpolation.primary, Qualifier::PerVertex);
}

TEST(VectorInterpolation, InterpMovP20UsesSecondVertexDelta) {
    const auto summary = TranslateInterpMov(1);

    EXPECT_EQ(summary.per_vertex_indices, (std::vector<u32>{0, 2}));
    EXPECT_EQ(summary.subtraction_count, 1u);
    EXPECT_EQ(summary.interpolation.primary, Qualifier::PerVertex);
}

TEST(VectorInterpolation, InterpMovP0UsesProvokingVertexValue) {
    const auto summary = TranslateInterpMov(2);

    EXPECT_EQ(summary.per_vertex_indices, (std::vector<u32>{0}));
    EXPECT_EQ(summary.subtraction_count, 0u);
    EXPECT_EQ(summary.interpolation.primary, Qualifier::PerVertex);
}

} // namespace
