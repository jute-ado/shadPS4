// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <optional>

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

} // namespace
