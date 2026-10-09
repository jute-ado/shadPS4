// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>

#include "gcn_test_runner.hpp"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/runtime_info.h"
#include "spirv/unified1/spirv.hpp"

namespace {
using namespace Shader;

constexpr u32 LowMask = 0x80000005U;
constexpr u32 HighMask = 0x8000000BU;
constexpr u32 Addend = 7U;
constexpr u32 Sentinel = 0xDEADBEEFU;

struct MbcntRecord {
    u32 rank;
    u32 native_lane;
};
using MbcntOutput = std::array<MbcntRecord, 64>;

std::vector<u32> EmitDivergentMbcntProbe(std::array<u32, 3> shape, bool select_upper_half) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.flattened_ud_buf.resize(4);
    const auto descriptor = AmdGpu::Buffer::Null();
    std::memcpy(info.flattened_ud_buf.data(), &descriptor, sizeof(descriptor));
    info.buffers.push_back({
        .sharp_idx = 0,
        .used_types = IR::Type::U32,
        .buffer_type = BufferType::Guest,
        .is_written = true,
    });
    IR::Program program{info};
    Pools pools{};
    auto* const entry = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {entry, body, merge};
    entry->AddBranch(body);
    entry->AddBranch(merge);
    body->AddBranch(merge);
    const auto append_block = [&](IR::Block* block) {
        auto& node = program.syntax_list.emplace_back();
        node.type = IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
    };
    append_block(entry);

    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Compute);
    runtime.cs_info.workgroup_size = shape;
    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 32 * 1024;
    Gcn::Translator translator{info, runtime, profile};
    translator.EmitPrologue(entry);
    IR::IREmitter ir{*entry};
    const auto x = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const auto y = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1);
    const auto z = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 2);
    const auto invocation = ir.IAdd(ir.IAdd(x, ir.IMul(y, ir.Imm32(shape[0]))),
                                    ir.IMul(z, ir.Imm32(shape[0] * shape[1])));
    const auto record = ir.IMul(invocation, ir.Imm32(2));
    ir.StoreBufferU32(1, ir.Imm32(0), record, ir.Imm32(Sentinel), {});
    ir.StoreBufferU32(1, ir.Imm32(0), ir.IAdd(record, ir.Imm32(1)), ir.LaneId(), {});
    ir.SetScalarReg(IR::ScalarReg::S4, ir.Imm32(LowMask));
    ir.SetScalarReg(IR::ScalarReg::S5, ir.Imm32(HighMask));
    ir.SetVectorReg(IR::VectorReg::V1, ir.Imm32(Addend));
    const auto upper = ir.IGreaterThanEqual(invocation, ir.Imm32(32), false);
    const auto predicate = select_upper_half ? upper : ir.LogicalNot(upper);
    ir.SetExec(predicate);
    const auto condition = ir.ConditionRef(predicate);
    auto& selection = program.syntax_list.emplace_back();
    selection.type = IR::AbstractSyntaxNode::Type::If;
    selection.data.if_node = {.cond = condition, .body = body, .merge = merge};
    append_block(body);

    Gcn::GcnInst high{};
    high.category = Gcn::InstCategory::VectorALU;
    high.opcode = Gcn::Opcode::V_MBCNT_HI_U32_B32;
    high.src[0].field = Gcn::OperandField::ScalarGPR;
    high.src[0].code = 5;
    high.src[1].field = Gcn::OperandField::VectorGPR;
    high.src[1].code = 1;
    high.dst[0].field = Gcn::OperandField::VectorGPR;
    high.dst[0].code = 1;
    auto low = high;
    low.opcode = Gcn::Opcode::V_MBCNT_LO_U32_B32;
    low.src[0].code = 4;
    const std::array instructions{high, low};
    translator.Translate(body, 0, instructions);
    IR::IREmitter body_ir{*body};
    body_ir.StoreBufferU32(1, body_ir.Imm32(0), record, body_ir.GetVectorReg(IR::VectorReg::V1),
                           {});
    auto& end = program.syntax_list.emplace_back();
    end.type = IR::AbstractSyntaxNode::Type::EndIf;
    end.data.end_if.merge = merge;
    append_block(merge);
    IR::IREmitter merge_ir{*merge};
    merge_ir.SetExec(merge_ir.Imm1(true));
    merge_ir.Epilogue();
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    Optimization::SsaRewritePass(program);
    Optimization::IdentityRemovalPass(program.blocks);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::DeadCodeEliminationPass(program);
    Optimization::CollectShaderInfoPass(program, profile);
    // Intentionally do not run the collective wave-lowering pass. MBCNT is local
    // arithmetic and must work in a partial-EXEC region without moving barriers.
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings);
}

class MbcntGpuConformance : public testing::Test {
protected:
    static void TearDownTestSuite() {
        gcn_test::Runner::DestroyInstance();
    }
};

TEST_F(MbcntGpuConformance, DivergentSingleWavePrefixKeepsNativeLaneControlForThreeHundredRuns) {
    auto result = gcn_test::Runner::subgroup32_instance();
    if (!result && result.error().code == gcn_test::Error::NoSuitableDevice) {
        GTEST_SKIP() << "MBCNT GPU proof unsupported, not executed: " << result.error().message;
    }
    ASSERT_TRUE(result.has_value()) << result.error().message;
    auto* const runner = *result;
    ASSERT_EQ(runner->effective_compute_subgroup_size(), 32U);
    u64 mismatches{};
    for (const auto shape :
         {std::array{64U, 1U, 1U}, std::array{8U, 8U, 1U}, std::array{4U, 4U, 4U}}) {
        for (const bool select_upper : {false, true}) {
            const auto spirv = EmitDivergentMbcntProbe(shape, select_upper);
            for (size_t word = 5; word < spirv.size();) {
                const auto count = spirv[word] >> 16;
                ASSERT_GT(count, 0U);
                ASSERT_LE(word + count, spirv.size());
                EXPECT_NE(spirv[word] & 0xFFFFU, u32(spv::Op::OpControlBarrier));
                word += count;
            }
            for (u32 repeat = 0; repeat < 50; ++repeat) {
                const auto output = runner->run<MbcntOutput>(spirv);
                ASSERT_TRUE(output.has_value()) << output.error().message;
                std::array<u32, 32> native_lane_counts{};
                for (u32 lane = 0; lane < 64; ++lane) {
                    const u64 below = (u64{1} << lane) - 1U;
                    const u64 mask = (u64{HighMask} << 32) | LowMask;
                    const u32 expected = ((lane >= 32U) == select_upper)
                                             ? Addend + u32(std::popcount(mask & below))
                                             : Sentinel;
                    mismatches += (*output)[lane].rank != expected;
                    if ((*output)[lane].native_lane < 32U) {
                        ++native_lane_counts[(*output)[lane].native_lane];
                    } else {
                        ++mismatches;
                    }
                }
                for (const auto count : native_lane_counts) {
                    mismatches += count != 2U;
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0U);
}
} // namespace
