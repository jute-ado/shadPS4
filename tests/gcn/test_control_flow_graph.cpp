// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "common/object_pool.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/frontend/structured_control_flow.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace {

Shader::Gcn::GcnInst Instruction(Shader::Gcn::Opcode opcode,
                                 Shader::Gcn::InstCategory category) {
    Shader::Gcn::GcnInst inst{};
    inst.opcode = opcode;
    inst.category = category;
    inst.length = 4;
    return inst;
}

std::vector<const Shader::IR::Inst*> Instructions(
    const Shader::IR::AbstractSyntaxList& syntax) {
    std::vector<const Shader::IR::Inst*> instructions;
    for (const auto& node : syntax) {
        if (node.type != Shader::IR::AbstractSyntaxNode::Type::Block) {
            continue;
        }
        for (const Shader::IR::Inst& inst : node.data.block->Instructions()) {
            instructions.push_back(&inst);
        }
    }
    return instructions;
}

std::vector<const Shader::IR::Inst*> InstructionsWithOpcode(
    const Shader::IR::AbstractSyntaxList& syntax, Shader::IR::Opcode opcode) {
    std::vector<const Shader::IR::Inst*> matches;
    for (const Shader::IR::Inst* inst : Instructions(syntax)) {
        if (inst->GetOpcode() == opcode) {
            matches.push_back(inst);
        }
    }
    return matches;
}

struct ScalarBranchCase {
    const char* name;
    Shader::Gcn::Opcode opcode;
    Shader::IR::Opcode raw_condition_opcode;
    bool zero_form;
};

class ComputeScalarBranchTest : public testing::TestWithParam<ScalarBranchCase> {};

} // namespace

TEST(ControlFlowGraph, ScalarGapBetweenExecMaskedRunsStaysUniform) {
    using namespace Shader::Gcn;

    std::array<GcnInst, 6> instructions{
        Instruction(Opcode::S_AND_SAVEEXEC_B64, InstCategory::ScalarALU),
        Instruction(Opcode::V_MOV_B32, InstCategory::VectorALU),
        Instruction(Opcode::S_MOV_B32, InstCategory::ScalarALU),
        Instruction(Opcode::V_MOV_B32, InstCategory::VectorALU),
        Instruction(Opcode::S_MOV_B64, InstCategory::ScalarALU),
        Instruction(Opcode::S_ENDPGM, InstCategory::FlowControl),
    };
    instructions[4].dst[0].field = OperandField::ExecLo;

    Common::ObjectPool<Block> block_pool{16};
    const CFG cfg{block_pool, instructions};

    std::vector<std::tuple<u32, u32, Shader::IR::Condition>> ranges;
    for (const Block& block : cfg.blocks) {
        ranges.emplace_back(block.begin_index, block.end_index, block.cond);
    }

    ASSERT_EQ(ranges.size(), 5U);
    EXPECT_EQ(ranges[0], std::make_tuple(0U, 0U, Shader::IR::Condition::Execnz));
    EXPECT_EQ(ranges[1], std::make_tuple(1U, 1U, Shader::IR::Condition::True));
    EXPECT_EQ(ranges[2], std::make_tuple(2U, 2U, Shader::IR::Condition::Execnz));
    EXPECT_EQ(ranges[3], std::make_tuple(3U, 3U, Shader::IR::Condition::True));
    EXPECT_EQ(ranges[4], std::make_tuple(4U, 5U, Shader::IR::Condition::True));
}

TEST(StructuredControlFlow, SyntheticExecPredicationScopeStaysLaneLocal) {
    using namespace Shader::Gcn;

    std::array<GcnInst, 6> instructions{
        Instruction(Opcode::S_AND_SAVEEXEC_B64, InstCategory::ScalarALU),
        Instruction(Opcode::V_MOV_B32, InstCategory::VectorALU),
        Instruction(Opcode::S_MOV_B32, InstCategory::ScalarALU),
        Instruction(Opcode::V_MOV_B32, InstCategory::VectorALU),
        Instruction(Opcode::S_MOV_B64, InstCategory::ScalarALU),
        Instruction(Opcode::S_ENDPGM, InstCategory::FlowControl),
    };
    instructions[4].dst[0].field = OperandField::ExecLo;

    Common::ObjectPool<Block> cfg_block_pool{16};
    CFG cfg{cfg_block_pool, instructions};
    // This test exercises compiler-created EXEC scopes only. Avoid translating the intentionally
    // operand-free fixture while retaining its CFG conditions.
    for (Block& block : cfg.blocks) {
        block.is_dummy = true;
    }
    Common::ObjectPool<Shader::IR::Inst> inst_pool{64};
    Common::ObjectPool<Shader::IR::Block> ir_block_pool{16};
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    Shader::Profile profile{};

    const auto syntax = BuildASL(inst_pool, ir_block_pool, cfg, info, runtime_info, profile);
    EXPECT_FALSE(InstructionsWithOpcode(syntax, Shader::IR::Opcode::GetExec).empty());
    EXPECT_TRUE(InstructionsWithOpcode(syntax, Shader::IR::Opcode::GuestWaveAny).empty());
}

TEST_P(ComputeScalarBranchTest, ReducesRawBitBeforeZeroFormInversion) {
    using namespace Shader::Gcn;
    const ScalarBranchCase& branch_case = GetParam();

    std::array<GcnInst, 3> instructions{
        Instruction(Opcode::S_NOP, InstCategory::FlowControl),
        Instruction(branch_case.opcode, InstCategory::FlowControl),
        Instruction(Opcode::S_ENDPGM, InstCategory::FlowControl),
    };
    // Make the scalar branch a loop back-edge. Unlike a forward conditional, this does not make
    // the structurer introduce an unrelated LogicalNot around the guest branch condition.
    instructions[1].control.sopp.simm = -2;

    Common::ObjectPool<Block> cfg_block_pool{16};
    CFG cfg{cfg_block_pool, instructions};
    Common::ObjectPool<Shader::IR::Inst> inst_pool{64};
    Common::ObjectPool<Shader::IR::Block> ir_block_pool{16};
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    Shader::Profile profile{};

    const auto syntax = BuildASL(inst_pool, ir_block_pool, cfg, info, runtime_info, profile);
    const auto wave_votes = InstructionsWithOpcode(syntax, Shader::IR::Opcode::GuestWaveAny);
    ASSERT_EQ(wave_votes.size(), 1U);
    const Shader::IR::Inst* raw_condition = wave_votes.front()->Arg(0).TryInstRecursive();
    ASSERT_NE(raw_condition, nullptr);
    EXPECT_EQ(raw_condition->GetOpcode(), branch_case.raw_condition_opcode);

    size_t inverted_vote_count{};
    for (const Shader::IR::Inst* logical_not :
         InstructionsWithOpcode(syntax, Shader::IR::Opcode::LogicalNot)) {
        if (logical_not->Arg(0).TryInstRecursive() == wave_votes.front()) {
            ++inverted_vote_count;
        }
    }
    EXPECT_EQ(inverted_vote_count, branch_case.zero_form ? 1U : 0U);
}

INSTANTIATE_TEST_SUITE_P(
    ScalarConditions, ComputeScalarBranchTest,
    testing::Values(
        ScalarBranchCase{"VccZ", Shader::Gcn::Opcode::S_CBRANCH_VCCZ,
                         Shader::IR::Opcode::GetVcc, true},
        ScalarBranchCase{"VccNz", Shader::Gcn::Opcode::S_CBRANCH_VCCNZ,
                         Shader::IR::Opcode::GetVcc, false},
        ScalarBranchCase{"ExecZ", Shader::Gcn::Opcode::S_CBRANCH_EXECZ,
                         Shader::IR::Opcode::GetExec, true},
        ScalarBranchCase{"ExecNz", Shader::Gcn::Opcode::S_CBRANCH_EXECNZ,
                         Shader::IR::Opcode::GetExec, false}),
    [](const testing::TestParamInfo<ScalarBranchCase>& info) { return info.param.name; });

TEST(StructuredControlFlow, B64MaskProducerCanonicalizesSccBeforeCselect) {
    using namespace Shader::Gcn;

    std::array<GcnInst, 3> instructions{
        Instruction(Opcode::S_AND_B64, InstCategory::ScalarALU),
        Instruction(Opcode::S_CSELECT_B32, InstCategory::ScalarALU),
        Instruction(Opcode::S_ENDPGM, InstCategory::FlowControl),
    };
    // B64 mask values remain lane-local, but SCC is a scalar "any bit set" result. Exercise a
    // non-uniform mask producer followed immediately by an SCC consumer.
    instructions[0].src[0].field = OperandField::ExecLo;
    instructions[0].src[1].field = OperandField::VccLo;
    instructions[0].dst[0].field = OperandField::ScalarGPR;
    instructions[0].dst[0].code = 0;
    instructions[1].src[0].field = OperandField::ScalarGPR;
    instructions[1].src[0].code = 2;
    instructions[1].src[1].field = OperandField::ScalarGPR;
    instructions[1].src[1].code = 3;
    instructions[1].dst[0].field = OperandField::ScalarGPR;
    instructions[1].dst[0].code = 4;

    Common::ObjectPool<Block> cfg_block_pool{16};
    CFG cfg{cfg_block_pool, instructions};
    Common::ObjectPool<Shader::IR::Inst> inst_pool{96};
    Common::ObjectPool<Shader::IR::Block> ir_block_pool{16};
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    Shader::Profile profile{};

    const auto syntax = BuildASL(inst_pool, ir_block_pool, cfg, info, runtime_info, profile);
    const auto set_scc = InstructionsWithOpcode(syntax, Shader::IR::Opcode::SetScc);
    const auto get_scc = InstructionsWithOpcode(syntax, Shader::IR::Opcode::GetScc);
    const auto selects = InstructionsWithOpcode(syntax, Shader::IR::Opcode::SelectU32);
    const auto wave_votes = InstructionsWithOpcode(syntax, Shader::IR::Opcode::GuestWaveAny);
    const auto mask_writes =
        InstructionsWithOpcode(syntax, Shader::IR::Opcode::SetThreadBitScalarRegPair);

    ASSERT_EQ(set_scc.size(), 1U);
    ASSERT_EQ(get_scc.size(), 1U);
    ASSERT_EQ(selects.size(), 1U);
    ASSERT_EQ(wave_votes.size(), 1U);
    ASSERT_EQ(mask_writes.size(), 1U);

    const Shader::IR::Inst* cselect_scc = selects.front()->Arg(0).TryInstRecursive();
    ASSERT_NE(cselect_scc, nullptr);
    EXPECT_EQ(cselect_scc->GetOpcode(), Shader::IR::Opcode::GetScc);

    const Shader::IR::Inst* canonical_scc = set_scc.front()->Arg(0).TryInstRecursive();
    ASSERT_NE(canonical_scc, nullptr);
    EXPECT_EQ(canonical_scc, wave_votes.front());

    const auto emitted = Instructions(syntax);
    const auto vote_position = std::ranges::find(emitted, wave_votes.front());
    const auto set_position = std::ranges::find(emitted, set_scc.front());
    const auto get_position = std::ranges::find(emitted, get_scc.front());
    ASSERT_NE(vote_position, emitted.end());
    ASSERT_NE(set_position, emitted.end());
    ASSERT_NE(get_position, emitted.end());
    EXPECT_LT(vote_position, set_position);
    EXPECT_LT(set_position, get_position);

    const Shader::IR::Inst* raw_mask = wave_votes.front()->Arg(0).TryInstRecursive();
    ASSERT_NE(raw_mask, nullptr);
    EXPECT_EQ(raw_mask->GetOpcode(), Shader::IR::Opcode::LogicalAnd);

    // Canonicalizing SCC must not collapse the guest B64 destination itself to one host-wide bit.
    const Shader::IR::Inst* stored_mask = mask_writes.front()->Arg(1).TryInstRecursive();
    ASSERT_NE(stored_mask, nullptr);
    EXPECT_EQ(stored_mask, raw_mask);
}
