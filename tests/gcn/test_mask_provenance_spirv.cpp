// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#endif

#include <gtest/gtest.h>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/recompiler.h"
#include "spirv/unified1/spirv.hpp"

namespace {

using namespace Shader;

size_t CountOpcode(const std::vector<u32>& spirv, spv::Op opcode) {
    size_t count{};
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        EXPECT_GT(words, 0u);
        EXPECT_LE(offset + words, spirv.size());
        if (static_cast<spv::Op>(word & 0xffff) == opcode) {
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
        if (static_cast<spv::Op>(word & 0xffff) == spv::OpCapability && words >= 2 &&
            spirv[offset + 1] == static_cast<u32>(capability)) {
            return true;
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

std::vector<u32> EmitMixedScalarMaskRepresentation(bool merge_high_word = false) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;

    Pools pools{};
    IR::Program program{info};
    IR::Block* const entry = pools.block_pool.Create(pools.inst_pool);
    IR::Block* const mask_path = pools.block_pool.Create(pools.inst_pool);
    IR::Block* const merge = pools.block_pool.Create(pools.inst_pool);
    entry->AddBranch(mask_path);
    entry->AddBranch(merge);
    mask_path->AddBranch(merge);
    program.blocks = {entry, mask_path, merge};

    IR::IREmitter entry_ir{*entry};
    entry_ir.Prologue();
    const IR::ScalarReg target_reg = merge_high_word ? IR::ScalarReg::S2 : IR::ScalarReg::S0;
    entry_ir.SetScalarReg(target_reg, entry_ir.Imm32(0xA5A5A5A5));
    const IR::U1 condition =
        entry_ir.ConditionRef(entry_ir.IEqual(entry_ir.LaneId(), entry_ir.Imm32(0)));

    IR::IREmitter mask_ir{*mask_path};
    if (merge_high_word) {
        mask_ir.SetThreadBitScalarRegPair(IR::ScalarReg::S1, mask_ir.Imm1(true));
    } else {
        mask_ir.SetThreadBitScalarReg(IR::ScalarReg::S0, mask_ir.Imm1(true));
    }

    IR::IREmitter merge_ir{*merge};
    merge_ir.Reference(merge_ir.GetScalarReg(target_reg));
    merge_ir.Epilogue();

    auto& entry_node = program.syntax_list.emplace_back();
    entry_node.type = IR::AbstractSyntaxNode::Type::Block;
    entry_node.data.block = entry;
    auto& if_node = program.syntax_list.emplace_back();
    if_node.type = IR::AbstractSyntaxNode::Type::If;
    if_node.data.if_node.cond = condition;
    if_node.data.if_node.body = mask_path;
    if_node.data.if_node.merge = merge;
    auto& mask_node = program.syntax_list.emplace_back();
    mask_node.type = IR::AbstractSyntaxNode::Type::Block;
    mask_node.data.block = mask_path;
    auto& end_if_node = program.syntax_list.emplace_back();
    end_if_node.type = IR::AbstractSyntaxNode::Type::EndIf;
    end_if_node.data.end_if.merge = merge;
    auto& merge_node = program.syntax_list.emplace_back();
    merge_node.type = IR::AbstractSyntaxNode::Type::Block;
    merge_node.data.block = merge;
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 64;
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.cs_info.workgroup_size = {64, 1, 1};

    Optimization::SsaRewritePass(program);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::IdentityRemovalPass(program.blocks);
    Optimization::DeadCodeEliminationPass(program);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::CollectShaderInfoPass(program, profile);

    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

TEST(MaskProvenanceSpirv, MixedRepresentationUsesOneVisibleWaveConsistentBallot) {
    const auto spirv = EmitMixedScalarMaskRepresentation();

    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityGroupNonUniform));
    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityGroupNonUniformBallot));
    EXPECT_EQ(CountOpcode(spirv, spv::OpGroupNonUniformBallot), 1u);
    EXPECT_EQ(CountOpcode(spirv, spv::OpSelect), 1u);
    // The only selection/branch is the source program's if. Scalar-mask materialization itself
    // must not add divergent backend control around the subgroup collective.
    EXPECT_EQ(CountOpcode(spirv, spv::OpSelectionMerge), 1u);
    EXPECT_EQ(CountOpcode(spirv, spv::OpBranchConditional), 1u);

#ifdef SHADPS4_SPIRV_VAL_PATH
    const auto module =
        std::filesystem::temp_directory_path() / "shadps4-mask-provenance-mixed.spv";
    {
        std::ofstream file{module, std::ios::binary | std::ios::trunc};
        ASSERT_TRUE(file.good());
        file.write(reinterpret_cast<const char*>(spirv.data()),
                   static_cast<std::streamsize>(spirv.size() * sizeof(u32)));
    }
    EXPECT_EQ(RunSpirvValidator(module), 0);
    std::error_code remove_error;
    std::filesystem::remove(module, remove_error);
#endif
}

TEST(MaskProvenanceSpirv, MixedHighWordProvenanceCombinesBallotWordsWithoutDivergentControl) {
    const auto spirv = EmitMixedScalarMaskRepresentation(/*merge_high_word=*/true);

    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityGroupNonUniform));
    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityGroupNonUniformBallot));
    EXPECT_EQ(CountOpcode(spirv, spv::OpGroupNonUniformBallot), 1u);
    EXPECT_EQ(CountOpcode(spirv, spv::OpBitwiseOr), 1u);
    EXPECT_EQ(CountOpcode(spirv, spv::OpSelectionMerge), 1u);
    EXPECT_EQ(CountOpcode(spirv, spv::OpBranchConditional), 1u);

#ifdef SHADPS4_SPIRV_VAL_PATH
    const auto module =
        std::filesystem::temp_directory_path() / "shadps4-mask-provenance-high-word.spv";
    {
        std::ofstream file{module, std::ios::binary | std::ios::trunc};
        ASSERT_TRUE(file.good());
        file.write(reinterpret_cast<const char*>(spirv.data()),
                   static_cast<std::streamsize>(spirv.size() * sizeof(u32)));
    }
    EXPECT_EQ(RunSpirvValidator(module), 0);
    std::error_code remove_error;
    std::filesystem::remove(module, remove_error);
#endif
}

} // namespace
