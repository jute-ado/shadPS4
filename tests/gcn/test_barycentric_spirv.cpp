// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include <gtest/gtest.h>

#include "common/object_pool.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
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

std::vector<u32> EmitMsaaArrayImageShader() {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.flattened_ud_buf.resize(sizeof(AmdGpu::Image) / sizeof(u32));

    auto image = AmdGpu::Image::Null(false);
    image.type = static_cast<u64>(AmdGpu::ImageType::Color2DMsaaArray);
    image.width = 63;
    image.height = 63;
    image.pitch = 63;
    image.depth = 3;
    image.last_array = 3;
    image.last_level = 1;
    std::memcpy(info.flattened_ud_buf.data(), &image, sizeof(image));
    info.images.push_back(ImageResource{
        .sharp_idx = 0,
        .is_array = true,
        .uses_integer_coordinates = true,
    });

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
    IR::TextureInstInfo texture_info{};
    texture_info.is_array.Assign(true);
    const auto coords = ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U));
    const auto texel = ir.ImageRead(ir.Imm32(0U), coords, {}, ir.Imm32(0U), texture_info);
    const auto dimensions =
        ir.ImageQueryDimension(ir.Imm32(0U), ir.Imm32(0U), ir.Imm1(false), texture_info);
    ir.Reference(texel);
    ir.Reference(dimensions);
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

std::vector<u32> EmitLaneIdShader(Stage stage, LogicalStage logical_stage, u32 subgroup_size = 32) {
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
    profile.subgroup_size = subgroup_size;

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(stage);
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

bool HasMsaaArrayImageType(const std::vector<u32>& spirv) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::OpTypeImage && words >= 9 &&
            spirv[offset + 3] == static_cast<u32>(spv::Dim2D) && spirv[offset + 5] == 1U &&
            spirv[offset + 6] == 1U) {
            return true;
        }
        offset += words;
    }
    return false;
}

std::optional<spv::StorageClass> FindAccessChainStorageClass(
    const std::vector<u32>& spirv, spv::StorageClass base_storage_class) {
    std::unordered_map<u32, spv::StorageClass> pointer_storage_classes;
    std::unordered_set<u32> matching_variables;
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        const auto opcode = static_cast<spv::Op>(word & 0xffff);
        if (opcode == spv::OpTypePointer && words >= 4) {
            pointer_storage_classes.emplace(
                spirv[offset + 1], static_cast<spv::StorageClass>(spirv[offset + 2]));
        } else if (opcode == spv::OpVariable && words >= 4 &&
                   spirv[offset + 3] == static_cast<u32>(base_storage_class)) {
            matching_variables.emplace(spirv[offset + 2]);
        }
        offset += words;
    }

    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        const auto opcode = static_cast<spv::Op>(word & 0xffff);
        if (opcode == spv::OpAccessChain && words >= 4 &&
            matching_variables.contains(spirv[offset + 3])) {
            const auto pointer_type = pointer_storage_classes.find(spirv[offset + 1]);
            if (pointer_type != pointer_storage_classes.end()) {
                return pointer_type->second;
            }
        }
        offset += words;
    }
    return std::nullopt;
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

TEST(BarycentricSpirv, FragCoordScaleUsesPushConstantPointerType) {
    const auto spirv = EmitNonAmdBarycentricShader(IR::Attribute::FragCoord);

    const auto storage_class =
        FindAccessChainStorageClass(spirv, spv::StorageClassPushConstant);
    ASSERT_TRUE(storage_class.has_value());
    EXPECT_EQ(*storage_class, spv::StorageClassPushConstant);
}

TEST(MsaaArraySpirv, DeclaresArrayedMultisampledImageAndEmitsFetchAndSizeQueries) {
    const auto spirv = EmitMsaaArrayImageShader();

    EXPECT_TRUE(HasMsaaArrayImageType(spirv));
    EXPECT_EQ(CountOpcode(spirv, spv::OpImageFetch), 1U);
    EXPECT_EQ(CountOpcode(spirv, spv::OpImageQuerySamples), 1U);
    EXPECT_EQ(CountOpcode(spirv, spv::OpImageQuerySize), 1U);
}

TEST(BufferAddressSpirv, GraphicsStagesExposeLaneIdForAddThreadIdAddressing) {
    for (const auto [stage, logical_stage] :
         {std::pair{Stage::Vertex, LogicalStage::Vertex},
          std::pair{Stage::Fragment, LogicalStage::Fragment}}) {
        const auto spirv = EmitLaneIdShader(stage, logical_stage);

        EXPECT_TRUE(HasCapability(spirv, spv::CapabilityGroupNonUniform));
        EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
    }
}

TEST(BufferAddressSpirv, ComputeLaneIdUsesWorkgroupIndexWhenHostSubgroupsAreNarrow) {
    const auto spirv = EmitLaneIdShader(Stage::Compute, LogicalStage::Compute, 32);

    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInLocalInvocationIndex));
    EXPECT_FALSE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
    EXPECT_GE(CountOpcode(spirv, spv::OpBitwiseAnd), 1U);
}

TEST(BufferAddressSpirv, ComputeLaneIdUsesNativeLaneWhenHostSupportsWave64) {
    const auto spirv = EmitLaneIdShader(Stage::Compute, LogicalStage::Compute, 64);

    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
    EXPECT_FALSE(HasBuiltIn(spirv, spv::BuiltInLocalInvocationIndex));
}

} // namespace
