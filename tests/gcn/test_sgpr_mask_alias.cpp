// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "gcn_test_runner.hpp"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/runtime_info.h"

namespace {
using namespace Shader;

struct MaskAliasCode {
    std::vector<u32> words;

    void Scalar(u32 destination, u32 value) {
        words.push_back(0xBE8000FFU | (destination << 16U) |
                        (static_cast<u32>(Gcn::OpcodeSOP1::S_MOV_B32) << 8U));
        words.push_back(value);
    }

    void Compare(u32 limit = 8U, bool set_exec = false, u32 destination = 4U) {
        // Synthetic V_CMP_LT_U32 s[4:5], v0, 8. The numeric register pair initially
        // contains unrelated values; the comparison must replace both halves.
        words.push_back(0xD0000000U |
                        (static_cast<u32>(set_exec ? Gcn::OpcodeVOP3::V_CMPX_LT_U32
                                                   : Gcn::OpcodeVOP3::V_CMP_LT_U32)
                         << 17U) |
                        destination);
        words.push_back(256U | ((128U + limit) << 9U));
    }

    void CopyPair(u32 destination, u32 source) {
        words.push_back(0xBE800000U | (destination << 16U) |
                        (static_cast<u32>(Gcn::OpcodeSOP1::S_MOV_B64) << 8U) | source);
    }

    void Count() {
        // V_BCNT_U32_B32 v1, s5, 0; V_BCNT_U32_B32 v1, s4, v1.
        words.push_back(0xD0000000U | (static_cast<u32>(Gcn::OpcodeVOP3::V_BCNT_U32_B32) << 17U) |
                        1U);
        words.push_back(5U | (128U << 9U));
        words.push_back((static_cast<u32>(Gcn::OpcodeVOP2::V_BCNT_U32_B32) << 25U) | (1U << 17U) |
                        (1U << 9U) | 4U);
    }

    void Prefix() {
        // V_MBCNT_HI_U32_B32 v2, s5, 0; V_MBCNT_LO_U32_B32 v2, s4, v2.
        words.push_back(0xD0000000U |
                        (static_cast<u32>(Gcn::OpcodeVOP3::V_MBCNT_HI_U32_B32) << 17U) | 2U);
        words.push_back(5U | (128U << 9U));
        words.push_back((static_cast<u32>(Gcn::OpcodeVOP2::V_MBCNT_LO_U32_B32) << 25U) |
                        (2U << 17U) | (2U << 9U) | 4U);
    }
};

struct MaskAliasFixture {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    RuntimeInfo runtime{};
    Profile profile{};
    IR::Block* block{};

    explicit MaskAliasFixture(const MaskAliasCode& code, u32 initial_active_lanes = 64U) {
        info.stage = Stage::Compute;
        info.l_stage = LogicalStage::Compute;
        runtime.Initialize(Stage::Compute);
        runtime.cs_info.workgroup_size = {64, 1, 1};
        block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        auto& node = program.syntax_list.emplace_back();
        node.type = IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
        program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
        program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

        Gcn::GcnCodeSlice slice{code.words.data(), code.words.data() + code.words.size()};
        Gcn::GcnDecodeContext decoder;
        std::vector<Gcn::GcnInst> instructions;
        while (!slice.atEnd()) {
            instructions.push_back(decoder.decodeInstruction(slice));
        }
        Gcn::Translator translator{info, runtime, profile};
        translator.Translate(block, 0U, instructions);
        // This fixture supplies only initial architectural state. All compare/move/count
        // instructions, their register writes, and their later SSA resolution are production.
        IR::IREmitter initial{*block, block->begin()};
        const auto lane = initial.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
        initial.SetExec(initial_active_lanes == 64U
                            ? initial.Imm1(true)
                            : initial.ILessThan(lane, initial.Imm32(initial_active_lanes), false));
        initial.SetVectorReg(IR::VectorReg::V0, lane);
    }

    void ReferenceNumericPairAndCount(u32 pair = 4U) {
        IR::IREmitter ir{*block};
        ir.Reference(ir.GetScalarReg<IR::U32>(IR::ScalarReg(pair)));
        ir.Reference(ir.GetScalarReg<IR::U32>(IR::ScalarReg(pair + 1U)));
        ir.Reference(ir.GetVectorReg<IR::U32>(IR::VectorReg::V1));
    }

    void Cleanup(bool preserve_exec = false) {
        Optimization::SsaRewritePass(program);
        Optimization::ConstantPropagationPass(program.blocks);
        Optimization::IdentityRemovalPass(program.blocks);
        Optimization::DeadCodeEliminationPass(program, preserve_exec);
    }

    std::vector<IR::Value> References() const {
        std::vector<IR::Value> values;
        for (const auto& inst : *block) {
            if (inst.GetOpcode() == IR::Opcode::Reference) {
                values.push_back(inst.Arg(0).Resolve());
            }
        }
        return values;
    }
};

bool DependsOn(IR::Value value, IR::Opcode opcode, unsigned budget = 128) {
    const auto* inst = value.TryInstRecursive();
    if (!inst || budget == 0) {
        return false;
    }
    if (inst->GetOpcode() == opcode) {
        return true;
    }
    for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
        if (DependsOn(inst->Arg(arg), opcode, budget - 1)) {
            return true;
        }
    }
    return false;
}

// Evaluates only this fixture's final SSA arithmetic; native wave64 lowering is tested
// separately. Production still owns decoding, all register writes, and SSA resolution.
u32 Evaluate(IR::Value value, u32 lane, u32 component = 0, unsigned budget = 64) {
    value = value.Resolve();
    if (value.IsImmediate()) {
        return value.Type() == IR::Type::U1 ? static_cast<u32>(value.U1()) : value.U32();
    }
    const auto* inst = value.TryInstRecursive();
    if (!inst || budget == 0) {
        ADD_FAILURE() << "Unexpected empty or cyclic numeric mask expression";
        return 0;
    }
    const auto arg = [&](size_t index) { return Evaluate(inst->Arg(index), lane, 0, budget - 1); };
    switch (inst->GetOpcode()) {
    case IR::Opcode::GetAttributeU32:
        EXPECT_EQ(inst->Arg(0).Attribute(), IR::Attribute::LocalInvocationId);
        return inst->Arg(1).U32() == 0 ? lane : 0U;
    case IR::Opcode::ULessThan32:
        return arg(0) < arg(1);
    case IR::Opcode::LogicalAnd:
        return arg(0) && arg(1);
    case IR::Opcode::BitCount32:
        return static_cast<u32>(std::popcount(arg(0)));
    case IR::Opcode::IAdd32:
        return arg(0) + arg(1);
    case IR::Opcode::CompositeExtractU32x4:
        return Evaluate(inst->Arg(0), lane, inst->Arg(1).U32(), budget - 1);
    case IR::Opcode::Ballot: {
        u32 word{};
        if (component < 2) {
            for (u32 bit = 0; bit < 32; ++bit) {
                if (Evaluate(inst->Arg(0), component * 32U + bit, 0, budget - 1) != 0) {
                    word |= 1U << bit;
                }
            }
        }
        return word;
    }
    default:
        ADD_FAILURE() << "Unexpected opcode in numeric mask expression: "
                      << static_cast<u32>(inst->GetOpcode());
        return 0;
    }
}

MaskAliasCode ComparisonCode() {
    MaskAliasCode code;
    code.Scalar(4, 3U);
    code.Scalar(5, 7U);
    code.Compare();
    return code;
}

TEST(SgprMaskAlias, VectorCompareReplacesBothNumericRegisterWords) {
    auto code = ComparisonCode();
    code.Count();
    MaskAliasFixture fixture{code};
    fixture.ReferenceNumericPairAndCount();
    fixture.Cleanup();
    const auto values = fixture.References();
    ASSERT_EQ(values.size(), 3U);
    EXPECT_TRUE(DependsOn(values[0], IR::Opcode::Ballot));
    EXPECT_TRUE(DependsOn(values[1], IR::Opcode::Ballot));
}

TEST(SgprMaskAlias, DecodedVectorBitCountReadsTheCompareMaskNotOldScalarNumbers) {
    auto code = ComparisonCode();
    code.Count();
    MaskAliasFixture fixture{code};
    fixture.ReferenceNumericPairAndCount();
    fixture.Cleanup();
    const auto values = fixture.References();
    ASSERT_EQ(values.size(), 3U);
    // Before alias repair this folds to popcount(3) + popcount(7) == 5, unrelated
    // to the eight true lanes in the preceding synthetic comparison.
    EXPECT_FALSE(values[2].IsImmediate());
    EXPECT_TRUE(DependsOn(values[2], IR::Opcode::Ballot));
    EXPECT_TRUE(DependsOn(values[2], IR::Opcode::BitCount32));
}

TEST(SgprMaskAlias, LaterScalarOverwriteOfLowWordRemainsAuthoritative) {
    auto code = ComparisonCode();
    code.Scalar(4, 0xA5A5A5A5U);
    code.Count();
    MaskAliasFixture fixture{code};
    fixture.ReferenceNumericPairAndCount();
    fixture.Cleanup();
    const auto values = fixture.References();
    ASSERT_EQ(values.size(), 3U);
    ASSERT_TRUE(values[0].IsImmediate());
    EXPECT_EQ(values[0].U32(), 0xA5A5A5A5U);
}

TEST(SgprMaskAlias, LaterScalarOverwriteOfBothWordsPreservesNumericCounting) {
    auto code = ComparisonCode();
    code.Scalar(4, 0xAAAAAAAAU);
    code.Scalar(5, 0x55555555U);
    code.Count();
    MaskAliasFixture fixture{code};
    fixture.ReferenceNumericPairAndCount();
    fixture.Cleanup();
    const auto values = fixture.References();
    ASSERT_EQ(values.size(), 3U);
    ASSERT_TRUE(values[2].IsImmediate());
    EXPECT_EQ(values[2].U32(), 32U);
    EXPECT_FALSE(DependsOn(values[2], IR::Opcode::Ballot));
}

TEST(SgprMaskAlias, NumericOnlyRegisterCountingDoesNotAcquireMaskDependencies) {
    MaskAliasCode code;
    code.Scalar(4, 3);
    code.Scalar(5, 7);
    code.Count();
    MaskAliasFixture fixture{code};
    fixture.ReferenceNumericPairAndCount();
    fixture.Cleanup();
    const auto values = fixture.References();
    ASSERT_EQ(values.size(), 3U);
    ASSERT_TRUE(values[2].IsImmediate());
    EXPECT_EQ(values[2].U32(), 5U);
}

TEST(SgprMaskAlias, OrdinaryScalarPairCopyRetainsPointerWordsNotPredicateBallot) {
    MaskAliasCode code;
    code.Scalar(6, 0x12345000U);
    code.Scalar(7, 2U);
    code.CopyPair(4, 6);
    code.Count();
    MaskAliasFixture fixture{code};
    fixture.ReferenceNumericPairAndCount();
    fixture.Cleanup();
    const auto values = fixture.References();
    ASSERT_EQ(values.size(), 3U);
    ASSERT_TRUE(values[0].IsImmediate());
    ASSERT_TRUE(values[1].IsImmediate());
    EXPECT_EQ(values[0].U32(), 0x12345000U);
    EXPECT_EQ(values[1].U32(), 2U);
    EXPECT_FALSE(DependsOn(values[0], IR::Opcode::Ballot));
    EXPECT_FALSE(DependsOn(values[1], IR::Opcode::Ballot));
}

TEST(SgprMaskAlias, ComparisonMaskIncludesUpperGuestWaveHalf) {
    MaskAliasCode code;
    code.Scalar(4, 3U);
    code.Scalar(5, 7U);
    code.Compare(40U);
    code.Count();
    MaskAliasFixture fixture{code};
    fixture.ReferenceNumericPairAndCount();
    fixture.Cleanup();
    const auto values = fixture.References();
    ASSERT_EQ(values.size(), 3U);
    EXPECT_EQ(Evaluate(values[0], 0), 0xFFFFFFFFU);
    EXPECT_EQ(Evaluate(values[1], 0), 0xFFU);
    EXPECT_EQ(Evaluate(values[2], 0), 40U);
}

TEST(SgprMaskAlias, CapturesOldExecBeforeLaterRestoreForCompareAndCmpx) {
    for (const bool cmpx : {false, true}) {
        SCOPED_TRACE(cmpx);
        MaskAliasCode code;
        code.Scalar(4, 3U);
        code.Scalar(5, 7U);
        code.Compare(40U, cmpx);
        code.CopyPair(126U, 193U); // s_mov_b64 exec, -1
        code.Count();
        MaskAliasFixture fixture{code, 4U};
        fixture.ReferenceNumericPairAndCount();
        fixture.Cleanup();
        const auto values = fixture.References();
        ASSERT_EQ(values.size(), 3U);
        EXPECT_EQ(Evaluate(values[0], 0), 0xFU);
        EXPECT_EQ(Evaluate(values[1], 0), 0U);
        EXPECT_EQ(Evaluate(values[2], 0), 4U);
    }
}

TEST(SgprMaskAlias, CmpxNumericBallotPrecedesItsExecWriteMarker) {
    MaskAliasCode code;
    code.Compare(40U, true);
    MaskAliasFixture fixture{code};
    const IR::Inst* ballot{};
    bool exec_written_after_ballot{};
    for (const auto& inst : *fixture.block) {
        if (inst.GetOpcode() == IR::Opcode::Ballot) {
            EXPECT_EQ(ballot, nullptr);
            ballot = &inst;
        }
        if (ballot && inst.GetOpcode() == IR::Opcode::SetExec) {
            exec_written_after_ballot = true;
            EXPECT_EQ(inst.Arg(0), ballot->Arg(0));
        }
    }
    EXPECT_NE(ballot, nullptr);
    EXPECT_TRUE(exec_written_after_ballot);
}

TEST(SgprMaskAlias, AllValidScalarPairsIncludingFinalPairReceiveBothWords) {
    for (u32 pair : {0U, 4U, 12U, 100U, 102U}) {
        SCOPED_TRACE(pair);
        MaskAliasCode code;
        code.Compare(40U, false, pair);
        MaskAliasFixture fixture{code};
        fixture.ReferenceNumericPairAndCount(pair);
        fixture.Cleanup();
        const auto values = fixture.References();
        ASSERT_EQ(values.size(), 3U);
        EXPECT_EQ(Evaluate(values[0], 0), 0xFFFFFFFFU);
        EXPECT_EQ(Evaluate(values[1], 0), 0xFFU);
    }
}

TEST(SgprMaskAlias, VccNumericAliasesUseTheSameComparisonMask) {
    MaskAliasCode code;
    code.Compare(40U, false, 106U); // VCC destination in the VOP3 encoding.
    MaskAliasFixture fixture{code};
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.GetVccLo());
    ir.Reference(ir.GetVccHi());
    fixture.Cleanup();
    const auto values = fixture.References();
    ASSERT_EQ(values.size(), 2U);
    EXPECT_EQ(Evaluate(values[0], 0), 0xFFFFFFFFU);
    EXPECT_EQ(Evaluate(values[1], 0), 0xFFU);
}

int TranslateOutOfBoundsComparePair() {
    MaskAliasCode code;
    code.Compare(40U, false, 103U);
    MaskAliasFixture fixture{code};
    if (!fixture.info.translation_failed) {
        return 1;
    }
    for (const auto& inst : *fixture.block) {
        if (inst.GetOpcode() == IR::Opcode::Ballot ||
            inst.GetOpcode() == IR::Opcode::SetScalarRegister ||
            inst.GetOpcode() == IR::Opcode::SetThreadBitScalarReg) {
            return 2;
        }
    }
    return 0;
}

TEST(SgprMaskAlias, MalformedPairFailsClosedBeforeAnyNumericOrPredicateWrite) {
    EXPECT_EXIT(std::_Exit(TranslateOutOfBoundsComparePair()), ::testing::ExitedWithCode(0), "");
}

struct MaskAliasGpuRecord {
    u32 low;
    u32 high;
    u32 count;
    u32 prefix;
};
using MaskAliasGpuOutput = std::array<MaskAliasGpuRecord, 64>;

std::vector<u32> EmitNumericMaskProbe(bool widen_guest_wave, bool cmpx) {
    MaskAliasCode code;
    code.Scalar(4, 3U);
    code.Scalar(5, 7U);
    code.Compare(40U, cmpx);
    code.Count();
    code.Prefix();
    MaskAliasFixture fixture{code};
    fixture.profile.supported_spirv = 0x00010600;
    fixture.profile.subgroup_size = 32;
    fixture.profile.max_shared_memory_size = 32 * 1024;
    fixture.info.flattened_ud_buf.resize(4);
    const auto output_buffer = AmdGpu::Buffer::Null();
    std::memcpy(fixture.info.flattened_ud_buf.data(), &output_buffer, sizeof(output_buffer));
    fixture.info.buffers.push_back({
        .sharp_idx = 0,
        .used_types = IR::Type::U32,
        .buffer_type = BufferType::Guest,
        .is_written = true,
    });
    IR::IREmitter begin{*fixture.block, fixture.block->begin()};
    begin.Prologue();
    IR::IREmitter ir{*fixture.block};
    const auto lane = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const auto record_base = ir.IMul(lane, ir.Imm32(4U));
    const std::array values{
        ir.GetScalarReg<IR::U32>(IR::ScalarReg::S4),
        ir.GetScalarReg<IR::U32>(IR::ScalarReg::S5),
        ir.GetVectorReg<IR::U32>(IR::VectorReg::V1),
        ir.GetVectorReg<IR::U32>(IR::VectorReg::V2),
    };
    for (u32 index = 0; index < values.size(); ++index) {
        ir.StoreBufferU32(1, ir.Imm32(0U), ir.IAdd(record_base, ir.Imm32(index)), values[index],
                          {});
    }
    ir.Epilogue();
    fixture.Cleanup(true);
    if (widen_guest_wave) {
        Optimization::LowerLaneIdToLocalInvocationIndexPass(fixture.program, fixture.runtime,
                                                            fixture.profile);
    }
    Optimization::IdentityRemovalPass(fixture.program.blocks);
    Optimization::DeadCodeEliminationPass(fixture.program);
    Optimization::ConstantPropagationPass(fixture.program.post_order_blocks);
    Optimization::CollectShaderInfoPass(fixture.program, fixture.profile);
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.runtime, fixture.program, bindings);
}

class SgprMaskAliasGpu : public testing::Test {
protected:
    static void TearDownTestSuite() {
        gcn_test::Runner::DestroyInstance();
    }
};

TEST_F(SgprMaskAliasGpu, NativeSubgroup32ControlCannotSupplyTheHighGuestMaskWord) {
    auto runner_result = gcn_test::Runner::subgroup32_instance();
    if (!runner_result && runner_result.error().code == gcn_test::Error::NoSuitableDevice) {
        GTEST_SKIP() << runner_result.error().message;
    }
    ASSERT_TRUE(runner_result.has_value()) << runner_result.error().message;
    auto* const runner = *runner_result;
    ASSERT_EQ(runner->effective_compute_subgroup_size(), 32U);
    const auto output = runner->run<MaskAliasGpuOutput>(EmitNumericMaskProbe(false, false));
    ASSERT_TRUE(output.has_value()) << output.error().message;
    for (const auto& record : *output) {
        EXPECT_EQ(record.high, 0U);
        EXPECT_EQ(record.count, static_cast<u32>(std::popcount(record.low)));
        EXPECT_LE(record.count, 32U);
        EXPECT_NE(record.count, 40U);
    }
}

TEST_F(SgprMaskAliasGpu, DecodedCompareAndNumericCountsSpanBothNativeSubgroups) {
    auto runner_result = gcn_test::Runner::subgroup32_instance();
    if (!runner_result && runner_result.error().code == gcn_test::Error::NoSuitableDevice) {
        GTEST_SKIP() << runner_result.error().message;
    }
    ASSERT_TRUE(runner_result.has_value()) << runner_result.error().message;
    auto* const runner = *runner_result;
    ASSERT_EQ(runner->effective_compute_subgroup_size(), 32U);
    for (const bool cmpx : {false, true}) {
        SCOPED_TRACE(cmpx);
        const auto spirv = EmitNumericMaskProbe(true, cmpx);
        for (u32 repeat = 0; repeat < 100U; ++repeat) {
            const auto output = runner->run<MaskAliasGpuOutput>(spirv);
            ASSERT_TRUE(output.has_value()) << "repeat=" << repeat << ' ' << output.error().message;
            for (u32 lane = 0; lane < output->size(); ++lane) {
                SCOPED_TRACE(lane);
                const auto& record = (*output)[lane];
                EXPECT_EQ(record.low, 0xFFFFFFFFU);
                EXPECT_EQ(record.high, 0xFFU);
                EXPECT_EQ(record.count, 40U);
                EXPECT_EQ(record.prefix, std::min(lane, 40U));
            }
        }
    }
}
} // namespace
