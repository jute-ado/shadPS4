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

std::vector<u32> EmitPerVertexDeltaShader(const bool use_amd_extension) {
    Info info{};
    info.stage = Stage::Fragment;
    info.l_stage = LogicalStage::Fragment;
    info.loads.Set(IR::Attribute::Param3, 2);
    info.fs_interpolation[3] = {Qualifier::PerVertex, Qualifier::None};

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
    const IR::F32 p0 = ir.GetAttribute(IR::Attribute::Param3, 2, 0);
    const IR::F32 p1 = ir.GetAttribute(IR::Attribute::Param3, 2, 1);
    ir.Reference(ir.FPSub(p1, p0));
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.supports_amd_shader_explicit_vertex_parameter = use_amd_extension;
    profile.supports_fragment_shader_barycentric = !use_amd_extension;

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Fragment);
    runtime_info.fs_info.num_inputs = 4;
    runtime_info.fs_info.inputs[3] = {
        .param_index = 3,
        .is_default = false,
        .is_flat = true,
        .default_value = 0,
    };
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

bool HasDecoration(const std::vector<u32>& spirv, spv::Decoration decoration) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpDecorate && words >= 3 &&
            spirv[offset + 2] == static_cast<u32>(decoration)) {
            return true;
        }
        offset += words;
    }
    return false;
}

struct InterpMovIrSummary {
    std::vector<u32> per_vertex_indices;
    bool source_coordinates_match{true};
    std::size_t subtraction_count{};
    bool has_subtraction_vertices{};
    u32 subtraction_lhs_vertex{};
    u32 subtraction_rhs_vertex{};
    bool destination_is_subtraction{};
    bool destination_is_vertex{};
    u32 destination_vertex{};
    Info::Interpolation interpolation{};
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
    runtime_info.fs_info.num_inputs = 4;
    runtime_info.fs_info.inputs[3] = {
        .param_index = 3,
        .is_default = false,
        .is_flat = true,
        .default_value = 0,
    };
    program.info.fs_interpolation[3] = {Qualifier::Smooth, Qualifier::Sample};

    Gcn::Translator translator{program.info, runtime_info, profile};
    translator.EmitPrologue(block);
    Gcn::GcnInst instruction{};
    instruction.control.vintrp.attr = 3;
    instruction.control.vintrp.chan = 2;
    instruction.src[0].code = parameter;
    instruction.dst[0].field = Gcn::OperandField::VectorGPR;
    instruction.dst[0].code = 5;
    translator.V_INTERP_MOV_F32(instruction);

    InterpMovIrSummary summary{.interpolation = program.info.fs_interpolation[3]};
    const IR::Inst* subtraction{};
    const IR::Inst* destination{};
    for (const IR::Inst& inst : *block) {
        if (inst.GetOpcode() == IR::Opcode::GetAttribute) {
            summary.per_vertex_indices.push_back(inst.Arg(2).U32());
            summary.source_coordinates_match &=
                inst.Arg(0).Attribute() == IR::Attribute::Param3 && inst.Arg(1).U32() == 2;
        } else if (inst.GetOpcode() == IR::Opcode::FPSub32) {
            ++summary.subtraction_count;
            subtraction = &inst;
        } else if (inst.GetOpcode() == IR::Opcode::SetVectorRegister &&
                   inst.Arg(0).VectorReg() == IR::VectorReg::V5) {
            destination = &inst;
        }
    }
    std::ranges::sort(summary.per_vertex_indices);

    if (subtraction != nullptr) {
        const IR::Inst* const lhs = subtraction->Arg(0).TryInstRecursive();
        const IR::Inst* const rhs = subtraction->Arg(1).TryInstRecursive();
        if (lhs != nullptr && rhs != nullptr && lhs->GetOpcode() == IR::Opcode::GetAttribute &&
            rhs->GetOpcode() == IR::Opcode::GetAttribute) {
            summary.has_subtraction_vertices = true;
            summary.subtraction_lhs_vertex = lhs->Arg(2).U32();
            summary.subtraction_rhs_vertex = rhs->Arg(2).U32();
        }
    }

    if (destination != nullptr) {
        const IR::Inst* value = destination->Arg(1).TryInstRecursive();
        if (value != nullptr && value->GetOpcode() == IR::Opcode::BitCastU32F32) {
            value = value->Arg(0).TryInstRecursive();
        }
        summary.destination_is_subtraction = value == subtraction;
        if (value != nullptr && value->GetOpcode() == IR::Opcode::GetAttribute) {
            summary.destination_is_vertex = true;
            summary.destination_vertex = value->Arg(2).U32();
        }
    }
    return summary;
}

struct DefaultInterpMovIrSummary {
    std::size_t attribute_load_count{};
    bool has_immediate_destination{};
    f32 destination{};
};

DefaultInterpMovIrSummary TranslateDefaultInterpMov(const u32 parameter) {
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
    runtime_info.fs_info.num_inputs = 4;
    runtime_info.fs_info.inputs[3] = {
        .param_index = 3,
        .is_default = true,
        .is_flat = false,
        .default_value = 1,
    };

    Gcn::Translator translator{program.info, runtime_info, profile};
    translator.EmitPrologue(block);
    Gcn::GcnInst instruction{};
    instruction.control.vintrp.attr = 3;
    instruction.control.vintrp.chan = 3;
    instruction.src[0].code = parameter;
    instruction.dst[0].field = Gcn::OperandField::VectorGPR;
    instruction.dst[0].code = 5;
    translator.V_INTERP_MOV_F32(instruction);

    DefaultInterpMovIrSummary summary{};
    for (const IR::Inst& inst : *block) {
        if (inst.GetOpcode() == IR::Opcode::GetAttribute) {
            ++summary.attribute_load_count;
        } else if (inst.GetOpcode() == IR::Opcode::SetVectorRegister &&
                   inst.Arg(0).VectorReg() == IR::VectorReg::V5) {
            const IR::Inst* const bitcast = inst.Arg(1).TryInstRecursive();
            if (bitcast != nullptr && bitcast->GetOpcode() == IR::Opcode::BitCastU32F32 &&
                bitcast->Arg(0).IsImmediate()) {
                summary.has_immediate_destination = true;
                summary.destination = bitcast->Arg(0).F32();
            }
        }
    }
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

TEST(BarycentricSpirv, KhrPerVertexDeltaUsesArrayValuesAndSubtraction) {
    const auto spirv = EmitPerVertexDeltaShader(false);

    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityFragmentBarycentricKHR));
    EXPECT_TRUE(HasDecoration(spirv, spv::DecorationPerVertexKHR));
    EXPECT_EQ(CountOpcode(spirv, spv::OpFSub), 1u);
    EXPECT_EQ(CountOpcode(spirv, spv::OpExtInst), 0u);
}

TEST(BarycentricSpirv, AmdPerVertexDeltaUsesExplicitRawValuesAndSubtraction) {
    const auto spirv = EmitPerVertexDeltaShader(true);

    EXPECT_TRUE(HasDecoration(spirv, spv::DecorationExplicitInterpAMD));
    EXPECT_EQ(CountOpcode(spirv, spv::OpFSub), 1u);
    EXPECT_EQ(CountOpcode(spirv, spv::OpExtInst), 3u);
}

TEST(VectorInterpolation, InterpMovP10UsesFirstVertexDelta) {
    const auto summary = TranslateInterpMov(0);

    EXPECT_EQ(summary.per_vertex_indices, (std::vector<u32>{0, 1}));
    EXPECT_TRUE(summary.source_coordinates_match);
    EXPECT_EQ(summary.subtraction_count, 1u);
    EXPECT_TRUE(summary.has_subtraction_vertices);
    EXPECT_EQ(summary.subtraction_lhs_vertex, 1u);
    EXPECT_EQ(summary.subtraction_rhs_vertex, 0u);
    EXPECT_TRUE(summary.destination_is_subtraction);
    EXPECT_EQ(summary.interpolation.primary, Qualifier::PerVertex);
    EXPECT_EQ(summary.interpolation.auxiliary, Qualifier::None);
}

TEST(VectorInterpolation, InterpMovP20UsesSecondVertexDelta) {
    const auto summary = TranslateInterpMov(1);

    EXPECT_EQ(summary.per_vertex_indices, (std::vector<u32>{0, 2}));
    EXPECT_TRUE(summary.source_coordinates_match);
    EXPECT_EQ(summary.subtraction_count, 1u);
    EXPECT_TRUE(summary.has_subtraction_vertices);
    EXPECT_EQ(summary.subtraction_lhs_vertex, 2u);
    EXPECT_EQ(summary.subtraction_rhs_vertex, 0u);
    EXPECT_TRUE(summary.destination_is_subtraction);
    EXPECT_EQ(summary.interpolation.primary, Qualifier::PerVertex);
    EXPECT_EQ(summary.interpolation.auxiliary, Qualifier::None);
}

TEST(VectorInterpolation, InterpMovP0UsesBaseVertexValue) {
    const auto summary = TranslateInterpMov(2);

    EXPECT_EQ(summary.per_vertex_indices, (std::vector<u32>{0}));
    EXPECT_TRUE(summary.source_coordinates_match);
    EXPECT_EQ(summary.subtraction_count, 0u);
    EXPECT_TRUE(summary.destination_is_vertex);
    EXPECT_EQ(summary.destination_vertex, 0u);
    EXPECT_EQ(summary.interpolation.primary, Qualifier::PerVertex);
    EXPECT_EQ(summary.interpolation.auxiliary, Qualifier::None);
}

TEST(VectorInterpolation, InterpMovRejectsInvalidParameterSelector) {
    EXPECT_DEATH(TranslateInterpMov(3), "");
}

TEST(VectorInterpolation, InterpMovDefaultP10AndP20AreZeroDeltas) {
    const auto p10 = TranslateDefaultInterpMov(0);
    const auto p20 = TranslateDefaultInterpMov(1);

    EXPECT_EQ(p10.attribute_load_count, 0u);
    EXPECT_TRUE(p10.has_immediate_destination);
    EXPECT_EQ(p10.destination, 0.0f);
    EXPECT_EQ(p20.attribute_load_count, 0u);
    EXPECT_TRUE(p20.has_immediate_destination);
    EXPECT_EQ(p20.destination, 0.0f);
}

TEST(VectorInterpolation, InterpMovDefaultP0UsesConfiguredComponent) {
    const auto p0 = TranslateDefaultInterpMov(2);

    EXPECT_EQ(p0.attribute_load_count, 0u);
    EXPECT_TRUE(p0.has_immediate_destination);
    EXPECT_EQ(p0.destination, 1.0f);
}

} // namespace
