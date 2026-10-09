// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <vector>
#include <gtest/gtest.h>

#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"

namespace {
using namespace Shader;

struct InitialExecFixture {
    explicit InitialExecFixture(std::array<u32, 3> shape, Stage physical = Stage::Compute,
                                LogicalStage logical = LogicalStage::Compute,
                                Stage runtime_stage = Stage::Compute)
        : program{info} {
        info.stage = physical;
        info.l_stage = logical;
        runtime.Initialize(runtime_stage);
        runtime.cs_info.workgroup_size = shape;
        block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        auto& node = program.syntax_list.emplace_back();
        node.type = IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
        program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
        program.post_order_blocks = IR::PostOrder(program.syntax_list.front());
        Gcn::Translator translator{info, runtime, profile};
        translator.EmitPrologue(block);
    }

    IR::Value InitialMask() const {
        const auto it = std::ranges::find_if(*block, [](const IR::Inst& inst) {
            return inst.GetOpcode() == IR::Opcode::SetExec;
        });
        EXPECT_NE(it, block->end());
        return it == block->end() ? IR::Value{} : it->Arg(0);
    }

    Info info{};
    IR::Program program;
    Pools pools{};
    RuntimeInfo runtime{};
    Profile profile{};
    IR::Block* block{};
};

std::optional<u32> EvaluateInitialMask(const IR::Value& value,
                                      const std::array<u32, 3>& local_id, u32 depth = 0) {
    if (depth > 24U) {
        return std::nullopt;
    }
    if (value.IsImmediate()) {
        return value.Type() == IR::Type::U1 ? u32(value.U1()) : value.U32();
    }
    const auto* const inst = value.TryInstRecursive();
    if (inst == nullptr) {
        return std::nullopt;
    }
    if (inst->GetOpcode() == IR::Opcode::GetAttributeU32) {
        if (inst->Arg(0).Attribute() != IR::Attribute::LocalInvocationId ||
            inst->Arg(1).U32() >= local_id.size()) {
            return std::nullopt;
        }
        return local_id[inst->Arg(1).U32()];
    }
    if (inst->NumArgs() != 2U) {
        return std::nullopt;
    }
    const auto lhs = EvaluateInitialMask(inst->Arg(0), local_id, depth + 1U);
    const auto rhs = EvaluateInitialMask(inst->Arg(1), local_id, depth + 1U);
    if (!lhs || !rhs) {
        return std::nullopt;
    }
    switch (inst->GetOpcode()) {
    case IR::Opcode::IAdd32:
        return *lhs + *rhs;
    case IR::Opcode::IMul32:
        return *lhs * *rhs;
    case IR::Opcode::ULessThan32:
        return u32(*lhs < *rhs);
    default:
        return std::nullopt;
    }
}

TEST(InitialComputeExec, SmallWorkgroupRetainsItsInitialActiveLaneBoundary) {
    for (const auto shape : std::array{std::array{1U, 1U, 1U}, std::array{32U, 1U, 1U},
                                      std::array{4U, 4U, 2U}, std::array{3U, 5U, 2U}}) {
        InitialExecFixture fixture{shape};
        const auto mask = fixture.InitialMask();
        EXPECT_FALSE(mask.IsImmediate());
        const u32 count = shape[0] * shape[1] * shape[2];
        for (u32 linear = 0; linear < 64U; ++linear) {
            SCOPED_TRACE(linear);
            const std::array local_id{linear % shape[0], (linear / shape[0]) % shape[1],
                                      linear / (shape[0] * shape[1])};
            const auto evaluated = EvaluateInitialMask(mask, local_id);
            ASSERT_TRUE(evaluated.has_value());
            EXPECT_EQ(*evaluated, u32(linear < count));
        }
    }
}

TEST(InitialComputeExec, FullOrInvalidWorkgroupsKeepTheirExistingPrologue) {
    constexpr u32 max = std::numeric_limits<u32>::max();
    for (const auto shape : std::array{std::array{64U, 1U, 1U}, std::array{8U, 8U, 1U},
                                      std::array{128U, 1U, 1U}, std::array{0U, 1U, 1U},
                                      std::array{1U, 0U, 32U}, std::array{max, max, max}}) {
        InitialExecFixture fixture{shape};
        const auto mask = fixture.InitialMask();
        ASSERT_TRUE(mask.IsImmediate());
        EXPECT_TRUE(mask.U1());
    }
}

TEST(InitialComputeExec, PhysicalLogicalAndRuntimeStagesMustAllBeCompute) {
    for (const u32 variant : {0U, 1U, 2U}) {
        InitialExecFixture fixture{{32U, 1U, 1U},
                                   variant == 0U ? Stage::Vertex : Stage::Compute,
                                   variant == 1U ? LogicalStage::Vertex : LogicalStage::Compute,
                                   variant == 2U ? Stage::Vertex : Stage::Compute};
        const auto mask = fixture.InitialMask();
        ASSERT_TRUE(mask.IsImmediate());
        EXPECT_TRUE(mask.U1());
    }
}

TEST(InitialComputeExec, SavedInitialMaskSurvivesSsaAndConstantCleanupDistinctFromFullExec) {
    InitialExecFixture fixture{{32U, 1U, 1U}};
    IR::IREmitter ir{*fixture.block};
    ir.SetThreadBitScalarReg(IR::ScalarReg::S4, ir.GetExec());
    ir.SetExec(ir.Imm1(true));
    ir.Reference(ir.GetExec());
    ir.SetExec(ir.GetThreadBitScalarReg(IR::ScalarReg::S4));
    ir.Reference(ir.GetExec());
    ir.Epilogue();
    Optimization::SsaRewritePass(fixture.program);
    Optimization::ConstantPropagationPass(fixture.program.post_order_blocks);
    Optimization::IdentityRemovalPass(fixture.program.blocks);
    Optimization::DeadCodeEliminationPass(fixture.program, true);
    std::vector<IR::Value> values;
    for (const auto& inst : *fixture.block) {
        if (inst.GetOpcode() == IR::Opcode::Reference) {
            values.push_back(inst.Arg(0));
        }
    }
    ASSERT_EQ(values.size(), 2U);
    ASSERT_TRUE(values[0].IsImmediate());
    EXPECT_TRUE(values[0].U1());
    EXPECT_FALSE(values[1].IsImmediate());
    EXPECT_EQ(EvaluateInitialMask(values[1], {0U, 0U, 1U}), 0U);
    EXPECT_EQ(EvaluateInitialMask(values[1], {0U, 0U, 0U}), 1U);
}

TEST(InitialComputeExec, RuntimeIdentityAlreadyDistinguishesWorkgroupDimensions) {
    RuntimeInfo small{};
    small.Initialize(Stage::Compute);
    small.cs_info.workgroup_size = {32U, 1U, 1U};
    auto full = small;
    full.cs_info.workgroup_size = {64U, 1U, 1U};
    EXPECT_NE(small, full);
}

} // namespace
