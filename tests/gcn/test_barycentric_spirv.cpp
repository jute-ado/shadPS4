// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <filesystem>
#include <fstream>

#ifdef _WIN32
#include <process.h>
#endif

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

std::vector<u32> EmitFragmentAttributeShader(IR::Attribute attribute, u32 component = 0) {
    Info info{};
    info.stage = Stage::Fragment;
    info.l_stage = LogicalStage::Fragment;
    info.loads.Set(attribute, component);

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
    ir.Reference(ir.GetAttribute(attribute, component));
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

bool HasPushConstantAccessChainWithPushConstantPointer(const std::vector<u32>& spirv) {
    const u32 id_bound = spirv.at(3);
    std::vector<u32> pointer_storage(id_bound, spv::StorageClassMax);
    u32 push_constant_variable = 0;

    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        const auto opcode = static_cast<spv::Op>(word & 0xffff);
        if (opcode == spv::OpTypePointer && words >= 4) {
            pointer_storage.at(spirv[offset + 1]) = spirv[offset + 2];
        } else if (opcode == spv::OpVariable && words >= 4 &&
                   spirv[offset + 3] == spv::StorageClassPushConstant) {
            push_constant_variable = spirv[offset + 2];
        }
        EXPECT_GT(words, 0u);
        offset += words;
    }

    if (push_constant_variable == 0) {
        return false;
    }
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        const auto opcode = static_cast<spv::Op>(word & 0xffff);
        if (opcode == spv::OpAccessChain && words >= 4 &&
            spirv[offset + 3] == push_constant_variable) {
            return pointer_storage.at(spirv[offset + 1]) == spv::StorageClassPushConstant;
        }
        offset += words;
    }
    return false;
}

#ifdef SHADPS4_SPIRV_VAL_PATH
int RunSpirvValidator(const std::filesystem::path& module) {
#ifdef _WIN32
    const std::string module_path = module.string();
    return static_cast<int>(_spawnl(_P_WAIT, SHADPS4_SPIRV_VAL_PATH, SHADPS4_SPIRV_VAL_PATH,
                                    "--target-env", "vulkan1.3", module_path.c_str(), nullptr));
#else
    const std::string command = std::string{"\""} + SHADPS4_SPIRV_VAL_PATH +
                                "\" --target-env vulkan1.3 \"" + module.string() + "\"";
    return std::system(command.c_str());
#endif
}
#endif

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

TEST(BarycentricSpirv, NonAmdSmoothUsesKhrInputDirectly) {
    const auto spirv = EmitFragmentAttributeShader(IR::Attribute::BaryCoordSmooth);

    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityFragmentBarycentricKHR));
    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityInterpolationFunction));
    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInBaryCoordKHR));
    EXPECT_EQ(CountGlslExtInst(spirv, GLSLstd450InterpolateAtCentroid), 0u);
    EXPECT_EQ(CountGlslExtInst(spirv, GLSLstd450InterpolateAtSample), 0u);
}

TEST(BarycentricSpirv, NonAmdCentroidUsesInterpolationFunction) {
    const auto spirv = EmitFragmentAttributeShader(IR::Attribute::BaryCoordSmoothCentroid);

    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityInterpolationFunction));
    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInBaryCoordKHR));
    EXPECT_EQ(CountGlslExtInst(spirv, GLSLstd450InterpolateAtCentroid), 1u);
}

TEST(BarycentricSpirv, NonAmdSampleUsesSampleIdAndSampleRateShading) {
    const auto spirv = EmitFragmentAttributeShader(IR::Attribute::BaryCoordSmoothSample);

    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityInterpolationFunction));
    EXPECT_TRUE(HasCapability(spirv, spv::CapabilitySampleRateShading));
    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInBaryCoordKHR));
    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInSampleId));
    EXPECT_EQ(CountGlslExtInst(spirv, GLSLstd450InterpolateAtSample), 1u);
}

TEST(FragmentInputSpirv, FragCoordHostScaleUsesPushConstantPointer) {
    const auto spirv = EmitFragmentAttributeShader(IR::Attribute::FragCoord);

    ASSERT_GE(spirv.size(), 5u);
    EXPECT_TRUE(HasPushConstantAccessChainWithPushConstantPointer(spirv));

#ifdef SHADPS4_SPIRV_VAL_PATH
    const auto output =
        std::filesystem::temp_directory_path() / "shadps4-fragcoord-push-constant-test.spv";
    {
        std::ofstream file{output, std::ios::binary | std::ios::trunc};
        ASSERT_TRUE(file.good());
        file.write(reinterpret_cast<const char*>(spirv.data()),
                   static_cast<std::streamsize>(spirv.size() * sizeof(u32)));
    }
    EXPECT_EQ(RunSpirvValidator(output), 0);
    std::error_code remove_error;
    std::filesystem::remove(output, remove_error);
#endif
}

} // namespace
