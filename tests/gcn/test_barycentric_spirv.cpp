// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <optional>
#include <unordered_map>

#include "common/object_pool.h"
#include "shader_recompiler/backend/spirv/spirv_composite_extract.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
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

bool HasDecorationOnBuiltIn(const std::vector<u32>& spirv, spv::BuiltIn builtin,
                            spv::Decoration decoration) {
    std::optional<u32> target;
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpDecorate && words >= 4 &&
            spirv[offset + 2] == static_cast<u32>(spv::DecorationBuiltIn) &&
            spirv[offset + 3] == static_cast<u32>(builtin)) {
            target = spirv[offset + 1];
            break;
        }
        offset += words;
    }
    if (!target) {
        return false;
    }
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpDecorate && words >= 3 &&
            spirv[offset + 1] == *target && spirv[offset + 2] == static_cast<u32>(decoration)) {
            return true;
        }
        offset += words;
    }
    return false;
}

bool HasOutOfBoundsVectorExtract(const std::vector<u32>& spirv) {
    std::unordered_map<u32, u32> vector_component_counts;
    std::unordered_map<u32, u32> result_types;

    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (words == 0 || words > spirv.size() - offset) {
            ADD_FAILURE() << "Malformed SPIR-V instruction while checking composite extracts";
            return true;
        }
        const auto opcode = static_cast<spv::Op>(word & 0xffff);
        if (opcode == spv::OpTypeVector && words >= 4) {
            vector_component_counts[spirv[offset + 1]] = spirv[offset + 3];
        } else if ((opcode == spv::OpConstantComposite || opcode == spv::OpCompositeConstruct) &&
                   words >= 3) {
            result_types[spirv[offset + 2]] = spirv[offset + 1];
        } else if (opcode == spv::OpCompositeExtract && words >= 5) {
            const auto result_type = result_types.find(spirv[offset + 3]);
            if (result_type != result_types.end()) {
                const auto vector_size = vector_component_counts.find(result_type->second);
                if (vector_size != vector_component_counts.end() &&
                    spirv[offset + 4] >= vector_size->second) {
                    return true;
                }
            }
        }
        offset += words;
    }
    return false;
}

std::vector<u32> EmitMismatchedF32x3ExtractShader(bool* translation_failed = nullptr) {
    Info info{};
    info.stage = Stage::Fragment;
    info.l_stage = LogicalStage::Fragment;

    IR::Program program{info};
    Pools pools{};
    IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    IR::IREmitter ir{*block};
    ir.Prologue();
    const IR::Value vector = ir.CompositeConstruct(ir.Imm32(1.0f), ir.Imm32(2.0f));
    // Construct malformed IR directly to exercise the backend's fail-closed boundary. The
    // public IREmitter::CompositeExtract API correctly rejects this index before it reaches here.
    const auto extract_it = block->PrependNewInst(block->end(), IR::Opcode::CompositeExtractF32x3,
                                                  {vector, ir.Imm32(2U)});
    ir.Reference(IR::Value{&*extract_it});
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Fragment);
    Backend::Bindings bindings{};
    auto spirv = Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
    if (translation_failed) {
        *translation_failed = info.translation_failed;
    }
    return spirv;
}

std::vector<u32> EmitMismatchedPackUfloatShader() {
    Info info{};
    info.stage = Stage::Fragment;
    info.l_stage = LogicalStage::Fragment;

    IR::Program program{info};
    Pools pools{};
    IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    IR::IREmitter ir{*block};
    ir.Prologue();
    const IR::Value vector = ir.CompositeConstruct(ir.Imm32(1.0f), ir.Imm32(2.0f));
    // PackUfloat10_11_11 is normally typed F32x3 by the frontend. Construct a malformed F32x2
    // operand directly to verify the backend does not emit extracts for missing components.
    const auto pack_it = block->PrependNewInst(block->end(), IR::Opcode::PackUfloat10_11_11,
                                               {vector});
    ir.Reference(IR::Value{&*pack_it});
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Fragment);
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

template <typename Callback>
void TranslateVInterpMov(u32 parameter, bool is_default, u8 default_value, bool is_flat,
                         Callback&& callback) {
    Info info{};
    info.stage = Stage::Fragment;
    info.l_stage = LogicalStage::Fragment;

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Fragment);
    runtime_info.fs_info.num_inputs = 1;
    runtime_info.fs_info.inputs[0] = {
        .param_index = 0,
        .is_default = is_default,
        .is_flat = is_flat,
        .default_value = default_value,
    };

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.supports_fragment_shader_barycentric = true;

    IR::Program program{info};
    Pools pools{};
    IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Gcn::Translator translator{program.info, runtime_info, profile};
    translator.EmitPrologue(block);

    Gcn::GcnInst inst{};
    inst.opcode = Gcn::Opcode::V_INTERP_MOV_F32;
    inst.src[0].code = parameter;
    inst.dst[0].field = Gcn::OperandField::VectorGPR;
    inst.dst[0].code = 0;
    inst.control.vintrp.attr = 0;
    inst.control.vintrp.chan = 2;
    translator.V_INTERP_MOV_F32(inst);

    callback(program.info, *block);
}

const IR::Inst* FindLastVectorRegisterWrite(const IR::Block& block) {
    for (auto it = block.rbegin(); it != block.rend(); ++it) {
        if (it->GetOpcode() == IR::Opcode::SetVectorRegister) {
            return &*it;
        }
    }
    return nullptr;
}

IR::Value GetVectorRegisterValue(const IR::Inst& write) {
    const IR::Value value = write.Arg(1).Resolve();
    const IR::Inst* const cast = value.TryInstRecursive();
    if (cast != nullptr && cast->GetOpcode() == IR::Opcode::BitCastU32F32) {
        return cast->Arg(0).Resolve();
    }
    return value;
}

void ExpectAttributeRead(const IR::Inst* inst, IR::Attribute attribute, u32 component,
                         u32 vertex_index) {
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(inst->GetOpcode(), IR::Opcode::GetAttribute);
    ASSERT_EQ(inst->NumArgs(), 3U);
    EXPECT_EQ(inst->Arg(0).Attribute(), attribute);
    EXPECT_EQ(inst->Arg(1).U32(), component);
    EXPECT_EQ(inst->Arg(2).U32(), vertex_index);
}

TEST(VectorInterpolation, InterpMovP10UsesFirstVertexDelta) {
    TranslateVInterpMov(0, false, 0, true, [](const Info&, const IR::Block& block) {
        const auto* const write = FindLastVectorRegisterWrite(block);
        ASSERT_NE(write, nullptr);
        const auto value = GetVectorRegisterValue(*write);
        const auto* const expression = value.TryInstRecursive();
        ASSERT_NE(expression, nullptr);
        ASSERT_EQ(expression->GetOpcode(), IR::Opcode::FPSub32);
        ExpectAttributeRead(expression->Arg(0).TryInstRecursive(), IR::Attribute::Param0, 2, 1);
        ExpectAttributeRead(expression->Arg(1).TryInstRecursive(), IR::Attribute::Param0, 2, 0);
    });
}

TEST(VectorInterpolation, InterpMovP20UsesSecondVertexDelta) {
    TranslateVInterpMov(1, false, 0, true, [](const Info&, const IR::Block& block) {
        const auto* const write = FindLastVectorRegisterWrite(block);
        ASSERT_NE(write, nullptr);
        const auto value = GetVectorRegisterValue(*write);
        const auto* const expression = value.TryInstRecursive();
        ASSERT_NE(expression, nullptr);
        ASSERT_EQ(expression->GetOpcode(), IR::Opcode::FPSub32);
        ExpectAttributeRead(expression->Arg(0).TryInstRecursive(), IR::Attribute::Param0, 2, 2);
        ExpectAttributeRead(expression->Arg(1).TryInstRecursive(), IR::Attribute::Param0, 2, 0);
    });
}

TEST(VectorInterpolation, InterpMovP0UsesFirstVertexValue) {
    TranslateVInterpMov(2, false, 0, false, [](const Info&, const IR::Block& block) {
        const auto* const write = FindLastVectorRegisterWrite(block);
        ASSERT_NE(write, nullptr);
        const auto value = GetVectorRegisterValue(*write);
        const auto* const expression = value.TryInstRecursive();
        ASSERT_NE(expression, nullptr);
        ExpectAttributeRead(expression, IR::Attribute::Param0, 2, 0);
    });
}

TEST(VectorInterpolation, InterpMovDefaultParametersUseZeroDeltasAndDefaultP0) {
    for (const u32 parameter : {0U, 1U}) {
        TranslateVInterpMov(parameter, true, 1, false, [](const Info&, const IR::Block& block) {
            const auto* const write = FindLastVectorRegisterWrite(block);
            ASSERT_NE(write, nullptr);
            const auto value = GetVectorRegisterValue(*write);
            ASSERT_TRUE(value.IsImmediate());
            EXPECT_FLOAT_EQ(value.F32(), 0.0f);
        });
    }
    TranslateVInterpMov(2, true, 2, false, [](const Info&, const IR::Block& block) {
        const auto* const write = FindLastVectorRegisterWrite(block);
        ASSERT_NE(write, nullptr);
        const auto value = GetVectorRegisterValue(*write);
        ASSERT_TRUE(value.IsImmediate());
        EXPECT_FLOAT_EQ(value.F32(), 1.0f);
    });
}

std::vector<u32> EmitLaneIdShader(Stage stage, LogicalStage logical_stage) {
    Info info{};
    info.stage = stage;
    info.l_stage = logical_stage;
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
    ir.Reference(ir.LaneId());
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(stage);
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

bool PushConstantAccessChainsUsePushConstantPointers(const std::vector<u32>& spirv) {
    const u32 id_bound = spirv.at(3);
    std::vector<std::optional<spv::StorageClass>> pointer_storage(id_bound);
    std::vector<std::optional<spv::StorageClass>> variable_storage(id_bound);

    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        const auto opcode = static_cast<spv::Op>(word & 0xffff);
        if (opcode == spv::OpTypePointer && words >= 4) {
            pointer_storage.at(spirv[offset + 1]) =
                static_cast<spv::StorageClass>(spirv[offset + 2]);
        } else if (opcode == spv::OpVariable && words >= 4) {
            variable_storage.at(spirv[offset + 2]) =
                static_cast<spv::StorageClass>(spirv[offset + 3]);
        }
        offset += words;
    }

    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        const auto opcode = static_cast<spv::Op>(word & 0xffff);
        if (opcode == spv::OpAccessChain && words >= 4) {
            const u32 result_type = spirv[offset + 1];
            const u32 base = spirv[offset + 3];
            if (variable_storage.at(base) == spv::StorageClassPushConstant &&
                pointer_storage.at(result_type) != spv::StorageClassPushConstant) {
                return false;
            }
        }
        offset += words;
    }
    return true;
}

TEST(BarycentricSpirv, NonAmdSmoothUsesKhrInputDirectly) {
    const auto spirv = EmitNonAmdBarycentricShader(IR::Attribute::BaryCoordSmooth);

    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityFragmentBarycentricKHR));
    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityInterpolationFunction));
    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInBaryCoordKHR));
    EXPECT_EQ(CountGlslExtInst(spirv, GLSLstd450InterpolateAtCentroid), 0u);
    EXPECT_EQ(CountGlslExtInst(spirv, GLSLstd450InterpolateAtSample), 0u);
}

TEST(SpirvCompositeExtract, MismatchedF32x3FromF32x2IndexIsNotEmitted) {
    bool translation_failed = false;
    const auto spirv = EmitMismatchedF32x3ExtractShader(&translation_failed);

    EXPECT_FALSE(HasOutOfBoundsVectorExtract(spirv));
    EXPECT_TRUE(translation_failed);
}

TEST(SpirvCompositeExtract, MismatchedPackUfloatDoesNotExtractMissingComponent) {
    const auto spirv = EmitMismatchedPackUfloatShader();
    EXPECT_FALSE(HasOutOfBoundsVectorExtract(spirv));
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpCompositeExtract), 0U);
}

TEST(SpirvCompositeExtract, IndexPolicyAcceptsOnlyComponentsWithinVector) {
    EXPECT_TRUE(Backend::SPIRV::IsValidCompositeExtractIndex(2, 0));
    EXPECT_TRUE(Backend::SPIRV::IsValidCompositeExtractIndex(2, 1));
    EXPECT_FALSE(Backend::SPIRV::IsValidCompositeExtractIndex(2, 2));
    EXPECT_FALSE(Backend::SPIRV::IsValidCompositeExtractIndex(2, 3));
    EXPECT_FALSE(Backend::SPIRV::IsValidCompositeExtractIndex(0, 0));
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

TEST(FragmentCoordinateSpirv, PushDataAccessUsesPushConstantPointer) {
    const auto spirv = EmitNonAmdBarycentricShader(IR::Attribute::FragCoord);

    EXPECT_TRUE(PushConstantAccessChainsUsePushConstantPointers(spirv));
}

TEST(SubgroupLaneIdSpirv, ComputeInputHasNoFragmentInterpolationDecoration) {
    const auto spirv = EmitLaneIdShader(Stage::Compute, LogicalStage::Compute);

    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
    EXPECT_FALSE(
        HasDecorationOnBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId, spv::DecorationFlat));
}

TEST(SubgroupLaneIdSpirv, FragmentInputPreservesFlatInterpolationDecoration) {
    const auto spirv = EmitLaneIdShader(Stage::Fragment, LogicalStage::Fragment);

    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
    EXPECT_TRUE(
        HasDecorationOnBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId, spv::DecorationFlat));
}

} // namespace
