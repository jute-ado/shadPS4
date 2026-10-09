// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

#include <gtest/gtest.h>
#include <half.hpp>

#include "gcn_test_runner.hpp"
#include "instructions.hpp"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "translator.hpp"
#include "spirv/unified1/spirv.hpp"

class GcnTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    static void TearDownTestSuite() {
        gcn_test::Runner::DestroyInstance();
    }
};

struct F32x2 {
    float a;
    float b;
};

namespace {

using namespace Shader::Gcn;

Shader::Gcn::GcnInst DecodeMimg(OpcodeMIMG opcode) {
    const u32 encoded_opcode = static_cast<u32>(opcode);
    const u64 instruction = static_cast<u64>(InstEncoding::MIMG) |
                            (static_cast<u64>(encoded_opcode & 0x7fU) << 18U) |
                            static_cast<u64>((encoded_opcode >> 7U) & 1U);
    const std::array words{static_cast<u32>(instruction), static_cast<u32>(instruction >> 32U)};
    Shader::Gcn::GcnCodeSlice slice{words.data(), words.data() + words.size()};
    Shader::Gcn::GcnDecodeContext decoder;
    return decoder.decodeInstruction(slice);
}

size_t CountSpirvOpcode(std::span<const u32> spirv, spv::Op opcode) {
    constexpr size_t SpirvHeaderWords = 5;
    size_t count{};
    for (size_t offset = SpirvHeaderWords; offset < spirv.size();) {
        const u32 instruction = spirv[offset];
        const size_t word_count = instruction >> 16;
        if (word_count == 0 || word_count > spirv.size() - offset) {
            return 0;
        }
        if ((instruction & 0xffffU) == static_cast<u32>(opcode)) {
            ++count;
        }
        offset += word_count;
    }
    return count;
}

size_t CountIrOpcode(const Shader::IR::Program& program, Shader::IR::Opcode opcode) {
    size_t count{};
    for (const Shader::IR::Block* const block : program.blocks) {
        count +=
            std::ranges::count_if(block->Instructions(), [opcode](const Shader::IR::Inst& inst) {
                return inst.GetOpcode() == opcode;
            });
    }
    return count;
}

bool BlockLocalDefinitionsPrecedeUses(const Shader::IR::Program& program) {
    for (const Shader::IR::Block* const block : program.blocks) {
        std::vector<const Shader::IR::Inst*> definitions;
        for (const Shader::IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() != Shader::IR::Opcode::Phi) {
                for (size_t index = 0; index < inst.NumArgs(); ++index) {
                    const Shader::IR::Value arg = inst.Arg(index);
                    if (arg.IsImmediate()) {
                        continue;
                    }
                    const Shader::IR::Inst* const definition = arg.Inst();
                    if (definition->GetParent() == block &&
                        std::ranges::find(definitions, definition) == definitions.end()) {
                        return false;
                    }
                }
            }
            definitions.push_back(&inst);
        }
    }
    return true;
}

} // namespace

TEST(GcnIrPass, SynchronizesDivergentWave64ReadFirstLaneInLaneRetiringLoop) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools;

    const auto make_block = [&] {
        Shader::IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        return block;
    };
    Shader::IR::Block* const entry = make_block();
    Shader::IR::Block* const outer_body = make_block();
    Shader::IR::Block* const loop_header = make_block();
    Shader::IR::Block* const loop_body = make_block();
    Shader::IR::Block* const read_block = make_block();
    Shader::IR::Block* const loop_continue = make_block();
    Shader::IR::Block* const loop_merge = make_block();
    Shader::IR::Block* const outer_merge = make_block();

    Shader::IR::IREmitter entry_ir{*entry};
    const Shader::IR::U32 invocation =
        entry_ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U1 outer_cond =
        entry_ir.ConditionRef(entry_ir.IGreaterThanEqual(invocation, entry_ir.Imm32(5U), false));
    Shader::IR::IREmitter{*outer_body}.Epilogue();
    Shader::IR::IREmitter{*loop_header}.Epilogue();

    Shader::IR::IREmitter body_ir{*loop_body};
    const Shader::IR::U1 lane_continues =
        body_ir.IGreaterThanEqual(invocation, body_ir.Imm32(10U), false);
    const Shader::IR::U1 break_cond = body_ir.ConditionRef(body_ir.LogicalNot(lane_continues));

    Shader::IR::IREmitter read_ir{*read_block};
    const Shader::IR::U32 first_active_lane = read_ir.ReadFirstLane(invocation);
    read_ir.Reference(first_active_lane);
    read_ir.Epilogue();

    Shader::IR::IREmitter continue_ir{*loop_continue};
    const Shader::IR::U1 repeat_cond = continue_ir.ConditionRef(continue_ir.Imm1(true));
    continue_ir.Epilogue();
    Shader::IR::IREmitter{*loop_merge}.Epilogue();
    Shader::IR::IREmitter{*outer_merge}.Epilogue();

    const auto add_block = [&](Shader::IR::Block* block) {
        auto& node = program.syntax_list.emplace_back();
        node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
    };
    add_block(entry);
    auto& outer_if = program.syntax_list.emplace_back();
    outer_if.type = Shader::IR::AbstractSyntaxNode::Type::If;
    outer_if.data.if_node = {.cond = outer_cond, .body = outer_body, .merge = outer_merge};
    add_block(outer_body);
    add_block(loop_header);
    auto& loop = program.syntax_list.emplace_back();
    loop.type = Shader::IR::AbstractSyntaxNode::Type::Loop;
    loop.data.loop = {.body = loop_body,
                      .continue_block = loop_continue,
                      .merge = loop_merge};
    add_block(loop_body);
    auto& break_node = program.syntax_list.emplace_back();
    break_node.type = Shader::IR::AbstractSyntaxNode::Type::Break;
    break_node.data.break_node = {.cond = break_cond, .merge = loop_merge, .skip = read_block};
    add_block(read_block);
    add_block(loop_continue);
    auto& repeat = program.syntax_list.emplace_back();
    repeat.type = Shader::IR::AbstractSyntaxNode::Type::Repeat;
    repeat.data.repeat = {.cond = repeat_cond,
                          .loop_header = loop_header,
                          .merge = loop_merge};
    add_block(loop_merge);
    auto& outer_end = program.syntax_list.emplace_back();
    outer_end.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
    outer_end.data.end_if.merge = outer_merge;
    add_block(outer_merge);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {64, 1, 1};

    Shader::Profile profile{};
    profile.max_shared_memory_size = 64 * 1024;
    profile.subgroup_size = 32;
    Shader::Optimization::ReadLaneWorkgroupPass(program, runtime_info, profile);

    ASSERT_TRUE(BlockLocalDefinitionsPrecedeUses(program));
    EXPECT_EQ(CountIrOpcode(program, Shader::IR::Opcode::ReadFirstLane), 0);
    EXPECT_GT(CountIrOpcode(program, Shader::IR::Opcode::Barrier), 0);
    EXPECT_GT(runtime_info.cs_info.shared_memory_size, 0U);

    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());
    profile.supported_spirv = 0x00010600;
    Shader::Optimization::CollectShaderInfoPass(program, profile);
    Shader::Backend::Bindings bindings{};
    const auto spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
    EXPECT_GT(CountSpirvOpcode(spirv, spv::Op::OpAtomicLoad), 0);
    EXPECT_GT(CountSpirvOpcode(spirv, spv::Op::OpAtomicExchange), 0);
    EXPECT_GT(CountSpirvOpcode(spirv, spv::Op::OpLoopMerge), 0);
    EXPECT_GE(CountSpirvOpcode(spirv, spv::Op::OpUGreaterThanEqual), 4);
}

TEST(GcnIrPass, SynchronizesSiblingNestedWave64LoopsWithMultipleRetirementBreaks) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools;

    const auto make_block = [&] {
        Shader::IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        return block;
    };
    const auto add_block = [&](Shader::IR::Block* block) {
        auto& node = program.syntax_list.emplace_back();
        node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
    };

    Shader::IR::Block* const entry = make_block();
    Shader::IR::Block* const outer_body = make_block();
    Shader::IR::Block* const outer_merge = make_block();
    Shader::IR::IREmitter entry_ir{*entry};
    const Shader::IR::U32 invocation =
        entry_ir.GetAttributeU32(Shader::IR::Attribute::LocalInvocationId, 0);
    const Shader::IR::U1 outer_cond =
        entry_ir.ConditionRef(entry_ir.IGreaterThanEqual(invocation, entry_ir.Imm32(1U), false));
    Shader::IR::IREmitter outer_ir{*outer_body};
    const Shader::IR::U1 first_inner_cond =
        outer_ir.ConditionRef(outer_ir.ILessThan(invocation, outer_ir.Imm32(32U), false));
    const Shader::IR::U1 second_inner_cond =
        outer_ir.ConditionRef(outer_ir.IGreaterThanEqual(invocation, outer_ir.Imm32(32U), false));
    outer_ir.Epilogue();
    Shader::IR::IREmitter{*outer_merge}.Epilogue();

    add_block(entry);
    auto& outer_if = program.syntax_list.emplace_back();
    outer_if.type = Shader::IR::AbstractSyntaxNode::Type::If;
    outer_if.data.if_node = {.cond = outer_cond, .body = outer_body, .merge = outer_merge};
    add_block(outer_body);

    const auto add_nested_loop = [&](const Shader::IR::U1 inner_cond) {
        Shader::IR::Block* const inner_body = make_block();
        Shader::IR::Block* const pre_read_body = make_block();
        Shader::IR::Block* const pre_read_merge = make_block();
        Shader::IR::Block* const loop_header = make_block();
        Shader::IR::Block* const loop_body = make_block();
        Shader::IR::Block* const read_block = make_block();
        Shader::IR::Block* const loop_continue = make_block();
        Shader::IR::Block* const loop_merge = make_block();
        Shader::IR::Block* const inner_merge = make_block();
        Shader::IR::IREmitter inner_body_ir{*inner_body};
        const Shader::IR::U1 pre_read_cond = inner_body_ir.ConditionRef(
            inner_body_ir.IGreaterThanEqual(invocation, inner_body_ir.Imm32(2U), false));
        inner_body_ir.Epilogue();
        Shader::IR::IREmitter pre_read_ir{*pre_read_body};
        const Shader::IR::U32 entry_first_lane = pre_read_ir.ReadFirstLane(invocation);
        pre_read_ir.Reference(entry_first_lane);
        pre_read_ir.Epilogue();
        Shader::IR::IREmitter{*pre_read_merge}.Epilogue();
        Shader::IR::IREmitter{*loop_header}.Epilogue();

        Shader::IR::IREmitter body_ir{*loop_body};
        const Shader::IR::U1 first_break = body_ir.ConditionRef(
            body_ir.ILessThan(invocation, body_ir.Imm32(4U), false));
        Shader::IR::IREmitter read_ir{*read_block};
        const Shader::IR::U32 first_lane = read_ir.ReadFirstLane(invocation);
        read_ir.Reference(first_lane);
        const Shader::IR::U1 second_break = read_ir.ConditionRef(
            read_ir.ILessThan(invocation, read_ir.Imm32(8U), false));
        Shader::IR::IREmitter continue_ir{*loop_continue};
        const Shader::IR::U1 repeat_cond = continue_ir.ConditionRef(continue_ir.Imm1(true));
        continue_ir.Epilogue();
        Shader::IR::IREmitter{*loop_merge}.Epilogue();
        Shader::IR::IREmitter{*inner_merge}.Epilogue();

        auto& inner_if = program.syntax_list.emplace_back();
        inner_if.type = Shader::IR::AbstractSyntaxNode::Type::If;
        inner_if.data.if_node = {.cond = inner_cond, .body = inner_body, .merge = inner_merge};
        add_block(inner_body);
        auto& pre_read_if = program.syntax_list.emplace_back();
        pre_read_if.type = Shader::IR::AbstractSyntaxNode::Type::If;
        pre_read_if.data.if_node = {
            .cond = pre_read_cond, .body = pre_read_body, .merge = pre_read_merge};
        add_block(pre_read_body);
        auto& pre_read_end = program.syntax_list.emplace_back();
        pre_read_end.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
        pre_read_end.data.end_if.merge = pre_read_merge;
        add_block(pre_read_merge);
        add_block(loop_header);
        auto& loop = program.syntax_list.emplace_back();
        loop.type = Shader::IR::AbstractSyntaxNode::Type::Loop;
        loop.data.loop = {.body = loop_body,
                          .continue_block = loop_continue,
                          .merge = loop_merge};
        add_block(loop_body);
        auto& first_break_node = program.syntax_list.emplace_back();
        first_break_node.type = Shader::IR::AbstractSyntaxNode::Type::Break;
        first_break_node.data.break_node = {
            .cond = first_break, .merge = loop_merge, .skip = read_block};
        add_block(read_block);
        auto& second_break_node = program.syntax_list.emplace_back();
        second_break_node.type = Shader::IR::AbstractSyntaxNode::Type::Break;
        second_break_node.data.break_node = {
            .cond = second_break, .merge = loop_merge, .skip = loop_continue};
        add_block(loop_continue);
        auto& repeat = program.syntax_list.emplace_back();
        repeat.type = Shader::IR::AbstractSyntaxNode::Type::Repeat;
        repeat.data.repeat = {
            .cond = repeat_cond, .loop_header = loop_header, .merge = loop_merge};
        add_block(loop_merge);
        auto& inner_end = program.syntax_list.emplace_back();
        inner_end.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
        inner_end.data.end_if.merge = inner_merge;
        add_block(inner_merge);
    };

    add_nested_loop(first_inner_cond);
    add_nested_loop(second_inner_cond);
    auto& outer_end = program.syntax_list.emplace_back();
    outer_end.type = Shader::IR::AbstractSyntaxNode::Type::EndIf;
    outer_end.data.end_if.merge = outer_merge;
    add_block(outer_merge);
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.workgroup_size = {64, 1, 1};
    Shader::Profile profile{};
    profile.max_shared_memory_size = 64 * 1024;
    profile.subgroup_size = 32;

    Shader::Optimization::ReadLaneWorkgroupPass(program, runtime_info, profile);

    EXPECT_EQ(CountIrOpcode(program, Shader::IR::Opcode::ReadFirstLane), 0);
    EXPECT_EQ(CountIrOpcode(program, Shader::IR::Opcode::Wave64ReadFirstLane), 4);
    EXPECT_GE(CountIrOpcode(program, Shader::IR::Opcode::Wave64LaneRetirementSync), 6);
}

TEST(GcnDecode, NeoMimgUsesTheExtendedOpcodeBitAndGradientModifier) {
    const auto instruction = DecodeMimg(OpcodeMIMG::IMAGE_SAMPLE_G16);
    const MimgModifierFlags modifiers{instruction.control.mimg.mod};

    EXPECT_EQ(instruction.opcode, Opcode::IMAGE_SAMPLE_G16);
    EXPECT_TRUE(modifiers.test(MimgModifier::Gradients));
    EXPECT_FALSE(modifiers.any(MimgModifier::Derivative, MimgModifier::LodClamp,
                               MimgModifier::Offset));
}

TEST(GcnDecode, NeoMimgPreservesBaseModifiersAlongsideGradientMode) {
    const auto instruction = DecodeMimg(OpcodeMIMG::IMAGE_SAMPLE_D_CL_O_G16);
    const MimgModifierFlags modifiers{instruction.control.mimg.mod};

    EXPECT_EQ(instruction.opcode, Opcode::IMAGE_SAMPLE_D_CL_O_G16);
    EXPECT_TRUE(modifiers.test(MimgModifier::Gradients));
    EXPECT_TRUE(modifiers.test(MimgModifier::Derivative));
    EXPECT_TRUE(modifiers.test(MimgModifier::LodClamp));
    EXPECT_TRUE(modifiers.test(MimgModifier::Offset));
}

TEST_F(GcnTest, mubuf_addr64_uses_vector_address) {
    // buffer_load_dword v0, v[0:1], s[4:7], 0 offset:12 addr64
    constexpr u64 addr64_load = 0x80010000e030800cULL;
    // Same instruction with ADDR64 clear. This descriptor-relative form must
    // not translate to the same memory access as the 64-bit VGPR form.
    constexpr u64 descriptor_relative_load = 0x80010000e030000cULL;

    const auto addr64_spirv = TranslateToSpirv(addr64_load);
    const auto descriptor_relative_spirv = TranslateToSpirv(descriptor_relative_load);

    EXPECT_NE(addr64_spirv, descriptor_relative_spirv);
}

TEST_F(GcnTest, mubuf_addr64_tracks_source_buffer_residency) {
    // The test epilogue writes through s[0:3]. Use a distinct source descriptor so this
    // assertion proves that ADDR64 itself keeps s[4:7] live for pre-draw residency.
    constexpr u64 addr64_load = 0x80010000e030800cULL;

    const auto result = TranslateToSpirvWithInfo(addr64_load, true);

    EXPECT_EQ(result.guest_buffer_count, 2U);
    EXPECT_EQ(result.addr64_pointer_root_count, 0U);
    EXPECT_FALSE(result.addr64_pointer_roots_overflow);
    EXPECT_TRUE(result.uses_dma);
}

TEST_F(GcnTest, mubuf_addr64_uses_unbounded_global_address_when_dma_is_enabled) {
    // Southern Islands ADDR64 adds the 64-bit VGPR address to the resource base but explicitly
    // ignores the resource size and performs no range checking. A bounded storage-buffer access
    // is therefore not equivalent: it turns a valid address beyond NUM_RECORDS into zero.
    constexpr u64 addr64_load = 0x80010000e030800cULL;

    const auto dma_enabled = TranslateToSpirvWithInfo(addr64_load, true);
    const auto dma_disabled = TranslateToSpirvWithInfo(addr64_load, false);

    EXPECT_EQ(dma_enabled.guest_buffer_count, 2U);
    EXPECT_EQ(dma_disabled.guest_buffer_count, 2U);
    EXPECT_TRUE(dma_enabled.uses_dma);
    EXPECT_FALSE(dma_disabled.uses_dma);
    EXPECT_NE(dma_enabled.spirv, dma_disabled.spirv);
}

TEST_F(GcnTest, mubuf_addr64_global_address_remains_runtime_rebindable) {
    constexpr u64 addr64_load = 0x80010000e030800cULL;

    const auto first = TranslateToSpirvWithInfo(addr64_load, true, 0x10000U);
    const auto rebound = TranslateToSpirvWithInfo(addr64_load, true, 0x24000U);

    EXPECT_EQ(first.guest_buffer_count, 2U);
    EXPECT_EQ(rebound.guest_buffer_count, 2U);
    // The descriptor base is read from live user data. Rebinding it must change the effective
    // absolute address without forcing a shader permutation.
    EXPECT_EQ(first.spirv, rebound.spirv);
}

// Example
// TEST_F(GcnTest, test_name) {
//     // Runner sets the vulkan context
//     auto runner = gcn_test::Runner::instance().value();
//
//     // v_add_f32 v0, v0, v1
//     auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_ADD_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
//
//     // run<T> tells how to interpret the result (only 32bit as of now)
//     // the second argument is templated, it can be at most 4 u32s
//     // the data is accessible by the instruction in v0-4 and s0-4 (mirrored)
//     // the result has to be placed in v0
//     auto result = runner->run<float>(spirv, F32x2{1.5f, 6.0f});
//
//     EXPECT_TRUE(result.has_value());
//     EXPECT_EQ(*result, 7.5f);
// }

TEST_F(GcnTest, add_f32) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_ADD_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<float>(spirv, F32x2{1.5f, 6.0f});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 7.5f);
}

TEST_F(GcnTest, add_i32_carry_feeds_addc_u32) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 2> instructions{
        VOP2(OpcodeVOP2::V_ADD_I32, VOperand8::V1, SOperand9::S0, VOperand8::V1).Get(),
        VOP2(OpcodeVOP2::V_ADDC_U32, VOperand8::V0, SOperand9::S2, VOperand8::V3).Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);

    const auto overflow = runner->run<u32>(spirv, std::array{0xffffffffU, 1U, 7U, 0U});
    ASSERT_TRUE(overflow.has_value());
    EXPECT_EQ(*overflow, 8U);

    const auto no_overflow = runner->run<u32>(spirv, std::array{2U, 1U, 7U, 0U});
    ASSERT_TRUE(no_overflow.has_value());
    EXPECT_EQ(*no_overflow, 7U);
}

bool HasSpirvOpcode(const std::vector<u32>& spirv, spv::Op opcode) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == opcode) {
            return true;
        }
        EXPECT_GT(words, 0u);
        offset += words;
    }
    return false;
}

u32 CountSpirvOpcode(const std::vector<u32>& spirv, spv::Op opcode) {
    u32 count{};
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        EXPECT_GT(words, 0u);
        count += static_cast<spv::Op>(word & 0xffff) == opcode;
        offset += words;
    }
    return count;
}

TEST_F(GcnTest, dx10_clamp_applies_once_at_vector_instruction_output) {
    // V_MAD_F32 lowers to a multiply and an add, but GCN applies DX10 NaN handling to the
    // completed VALU instruction result rather than every internal host operation.
    const auto mad =
        VOP3A(OpcodeVOP3::V_MAD_F32, VOperand8::V0, SOperand9::V0, SOperand9::V1,
              SOperand9::V2)
            .Get();

    const auto disabled = TranslateToSpirvWithDx10Clamp(mad, false);
    const auto enabled = TranslateToSpirvWithDx10Clamp(mad, true);

    EXPECT_EQ(CountSpirvOpcode(disabled, spv::Op::OpIsNan), 0U);
    EXPECT_EQ(CountSpirvOpcode(enabled, spv::Op::OpIsNan), 1U);
}

TEST_F(GcnTest, dx10_clamp_does_not_change_raw_bit_moves) {
    // V_MOV_B32 carries untyped payloads such as packed image offsets. Its translator uses an
    // F32 container, but the instruction itself has no floating-point semantics.
    const auto move = VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::V1).Get();

    const auto enabled = TranslateToSpirvWithDx10Clamp(move, true);

    EXPECT_EQ(CountSpirvOpcode(enabled, spv::Op::OpIsNan), 0U);
}

bool HasSpirvCapability(const std::vector<u32>& spirv, spv::Capability capability) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpCapability && words >= 2 &&
            spirv[offset + 1] == static_cast<u32>(capability)) {
            return true;
        }
        EXPECT_GT(words, 0u);
        offset += words;
    }
    return false;
}

TEST_F(GcnTest, compare_with_explicit_sgpr_destination_materializes_scalar_mask) {
    auto runner = gcn_test::Runner::instance().value();

    // Guest compilers may use a VOP3 compare to write an arbitrary SGPR pair and then inspect
    // its low word with scalar ALU instructions. Keep that scalar view coherent with the
    // per-invocation mask representation used for EXEC/VCC handling.
    const std::array<u64, 2> instructions{
        VOP3A(OpcodeVOP3::V_CMP_LG_U32, VOperand8::V14, SOperand9::Const0,
              SOperand9::V0)
            .Get(),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S14).Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);

    const auto false_mask = runner->run<u32>(spirv, u32{0});
    ASSERT_TRUE(false_mask.has_value());
    EXPECT_EQ(*false_mask, 0U);

    const auto true_mask = runner->run<u32>(spirv, u32{1});
    ASSERT_TRUE(true_mask.has_value());
    EXPECT_EQ(*true_mask, 1U);
}

TEST_F(GcnTest, scalar_mask_write_is_visible_to_vector_boolean_consumer) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 3> instructions{
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::S14, SOperand8::ConstNeg1).Get(),
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::S15, SOperand8::Const0).Get(),
        VOP3A(OpcodeVOP3::V_CNDMASK_B32, VOperand8::V0, SOperand9::Const0,
              SOperand9::Const1, SOperand9::S14)
            .Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);

    const auto result = runner->run<u32>(spirv, u32{0});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1U);
}

TEST_F(GcnTest, scalar_to_scalar_mov_b64_preserves_full_scalar_words) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 2> instructions{
        SOP1(OpcodeSOP1::S_MOV_B64, SOperand7::S14, SOperand8::S0).Get(),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S14).Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);

    constexpr u32 input = 0x12345678U;
    const auto result = runner->run<u32>(spirv, input);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, input);
}

TEST_F(GcnTest, overlapping_scalar_mov_b64_snapshots_both_source_words) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 4> instructions{
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::S14, SOperand8::Const1).Get(),
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::S15, SOperand8::ConstNeg1).Get(),
        SOP1(OpcodeSOP1::S_MOV_B64, SOperand7::S15, SOperand8::S14).Get(),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S16).Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);

    const auto result = runner->run<u32>(spirv, u32{0});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, std::numeric_limits<u32>::max());
}

TEST_F(GcnTest, saveexec_scc_reports_any_active_lane) {
    // SCC is scalar: the ISA defines it as (new EXEC != 0), not as the current lane's EXEC bit.
    // Keep SCC live through a scalar select so the generated shader must retain the subgroup vote.
    const std::array<u64, 3> instructions{
        SOP1(OpcodeSOP1::S_AND_SAVEEXEC_B64, SOperand7::S14, SOperand8::VccLo).Get(),
        SOP2(OpcodeSOP2::S_CSELECT_B32, SOperand7::S0, SOperand8::Const1,
             SOperand8::Const0)
            .Get(),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S0).Get(),
    };

    const auto spirv = TranslateToSpirv(instructions);
    EXPECT_TRUE(HasSpirvOpcode(spirv, spv::Op::OpGroupNonUniformAny));
    EXPECT_TRUE(HasSpirvCapability(spirv, spv::CapabilityGroupNonUniformVote));
}

TEST_F(GcnTest, scalar_data_survives_writelane_readlane_round_trip) {
    auto runner = gcn_test::Runner::instance().value();
    constexpr auto lane0 = static_cast<VOperand8>(128);
    const std::array<u64, 5> instructions{
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::S20, SOperand8::S0).Get(),
        VOP2(OpcodeVOP2::V_WRITELANE_B32, VOperand8::V29, SOperand9::S20, lane0).Get(),
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::S20, SOperand8::Const0).Get(),
        VOP2(OpcodeVOP2::V_READLANE_B32, VOperand8::V20, SOperand9::V29, lane0).Get(),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S20).Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);

    constexpr u32 input = 0x12345678U;
    const auto result = runner->run<u32>(spirv, input);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, input);
}

TEST_F(GcnTest, scalar_mask_survives_writelane_readlane_round_trip) {
    auto runner = gcn_test::Runner::instance().value();
    constexpr auto lane0 = static_cast<VOperand8>(128);
    const std::array<u64, 5> instructions{
        VOP3A(OpcodeVOP3::V_CMP_LG_U32, VOperand8::V20, SOperand9::Const0,
              SOperand9::V0)
            .Get(),
        VOP2(OpcodeVOP2::V_WRITELANE_B32, VOperand8::V29, SOperand9::S20, lane0).Get(),
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::S20, SOperand8::Const0).Get(),
        VOP2(OpcodeVOP2::V_READLANE_B32, VOperand8::V20, SOperand9::V29, lane0).Get(),
        VOP3A(OpcodeVOP3::V_CNDMASK_B32, VOperand8::V0, SOperand9::Const0,
              SOperand9::Const1, SOperand9::S20)
            .Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);

    const auto false_result = runner->run<u32>(spirv, u32{0});
    ASSERT_TRUE(false_result.has_value());
    EXPECT_EQ(*false_result, 0U);

    const auto true_result = runner->run<u32>(spirv, u32{1});
    ASSERT_TRUE(true_result.has_value());
    EXPECT_EQ(*true_result, 1U);
}

TEST_F(GcnTest, explicit_compare_mask_feeds_scalar_find_first_bit) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 3> instructions{
        VOP3A(OpcodeVOP3::V_CMP_LG_U32, VOperand8::V14, SOperand9::Const0,
              SOperand9::V0)
            .Get(),
        SOP1(OpcodeSOP1::S_FF1_I32_B32, SOperand7::S16, SOperand8::S14).Get(),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S16).Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);

    const auto no_lanes = runner->run<u32>(spirv, u32{0});
    ASSERT_TRUE(no_lanes.has_value());
    EXPECT_EQ(*no_lanes, std::numeric_limits<u32>::max());

    const auto first_lane = runner->run<u32>(spirv, u32{1});
    ASSERT_TRUE(first_lane.has_value());
    EXPECT_EQ(*first_lane, 0U);
}

TEST_F(GcnTest, resource_tracking_follows_scalar_mask_representation_select) {
    // The guest may reuse descriptor SGPRs for a temporary lane mask and restore them through a
    // scalar select before a buffer access. Resource tracking must still discover the user-data
    // descriptor source through the representation-select introduced by SSA reconciliation.
    constexpr u64 addr64_load = 0x80010000e030800cULL;
    const std::array<u64, 3> instructions{
        VOP3A(OpcodeVOP3::V_CMP_LG_U32, VOperand8::V8, SOperand9::Const0,
              SOperand9::V0)
            .Get(),
        SOP2(OpcodeSOP2::S_CSELECT_B32, SOperand7::S4, SOperand8::S4, SOperand8::S8).Get(),
        addr64_load,
    };

    const auto result = TranslateToSpirvWithInfo(instructions);
    EXPECT_FALSE(result.spirv.empty());
    EXPECT_EQ(result.guest_buffer_count, 2U);
}

TEST_F(GcnTest, add_nan) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_ADD_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<float>(spirv, F32x2{1.0f, std::numeric_limits<float>::quiet_NaN()});

    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(std::isnan(*result));
}

using half = half_float::half;

struct F16x2 {
    half a;
    half b = half(0.0f);

    bool operator==(const F16x2& rhs) const = default;
};

static_assert(sizeof(F16x2) == sizeof(float));

TEST_F(GcnTest, add_f16) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_ADD_F16, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f)}, F16x2{half(1.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, F16x2{half(2.0f)});
}

TEST_F(GcnTest, add_f16_clamp) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetClamp(true).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f)}, F16x2{half(1.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, F16x2{half(1.0f)}); //confirmed with neo
}

TEST_F(GcnTest, add_f16_neg) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetNeg({true, true, false}).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f)}, F16x2{half(1.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ((*result).a, half(-2.0f)); //confirmed with neo
}

TEST_F(GcnTest, add_f16_opsel_hi) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetOpSel({true, true, false, true}).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f), half(2.0f)}, F16x2{half(1.0f), half(2.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ((*result).a, half(1.0f));
    EXPECT_EQ((*result).b, half(4.0f));
}

TEST_F(GcnTest, sub_f16) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_SUB_F16, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(0.0f)}, F16x2{half(1.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, F16x2{half(-1.0f)}); //confirmed with neo
}

TEST_F(GcnTest, mul_legacy_nan) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_MUL_LEGACY_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<u32>(spirv, std::array{u32(0), u32(0x7fc00000)});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0);
}

TEST_F(GcnTest, mul_nan) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_MUL_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<float>(spirv, std::array{u32(0), u32(0x7fc00000)});

    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(std::isnan(*result));
}

TEST_F(GcnTest, min_legacy_nan) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_MIN_LEGACY_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<u32>(spirv, std::array{u32(0), u32(0x7fc00000)});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x7fc00000);
}

TEST_F(GcnTest, min_nan) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_MIN_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<float>(spirv, std::array{u32(0), u32(0x7fc00000)});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0);
}

TEST_F(GcnTest, add3_u32_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD3_U32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).Get());
    auto result = runner->run<u32>(spirv, std::array{0, 1, 2});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 3);
}

TEST_F(GcnTest, add3_u32_2) {
    auto runner = gcn_test::Runner::instance().value();
    auto big = 2000000000;

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD3_U32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).Get());
    auto result = runner->run<u32>(spirv, std::array{big, big, big});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x65A0BC00);
}

TEST_F(GcnTest, add3_u32_3) {
    auto runner = gcn_test::Runner::instance().value();
    auto big = 2000000000;

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD3_U32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetClamp(true).Get());
    auto result = runner->run<u32>(spirv, std::array{big, big, big});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x65A0BC00);
}

TEST_F(GcnTest, add3_u32_4) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD3_U32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetNeg({1,0,0}).Get());
    auto result = runner->run<u32>(spirv, std::array{0, 1, 2});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x80000003);
}

TEST_F(GcnTest, or3_u32_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_OR3_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0xF0F0F0F0, 0x07070707, 0x11111111});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xF7F7F7F7);
}

TEST_F(GcnTest, or3_u32_2) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_OR3_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).Get());
    auto result = runner->run<u32>(spirv, std::array{0x07070707, 0x11111111, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x57575757);
}

TEST_F(GcnTest, or3_u32_3) {
    auto runner = gcn_test::Runner::instance().value();
    auto big = 2000000000;

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_OR3_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetClamp(true).Get());
    auto result = runner->run<u32>(spirv, std::array{0x07070707, 0x11111111, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x57575757);
}

TEST_F(GcnTest, or3_u32_4) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_OR3_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetNeg({0,0,1}).Get());
    auto result = runner->run<u32>(spirv, std::array{0x07070707, 0x11111111, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xD7575757);
}

TEST_F(GcnTest, and_or_b32_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0xF0F0F0F0, 0x07070707, 0x11111111});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x11111111);
}

TEST_F(GcnTest, and_or_b32_2) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOmod(Omod::Mul2).Get());
    auto result = runner->run<u32>(spirv, std::array{0x40404040, 0x40404040, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x40404040);
}

TEST_F(GcnTest, and_or_b32_3) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetClamp(true).Get());
    auto result = runner->run<u32>(spirv, std::array{0x40404040, 0x40404040, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x40404040);
}

TEST_F(GcnTest, and_or_b32_4) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetNeg({1,0,0}).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0x07070707, 0x11111111, 0xF0F0F0F0});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xF1F1F1F1);
}

TEST_F(GcnTest, and_or_b32_5) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetNeg({1,0,0}).SetAbs({1,0,0}).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0x77777777, 0xB0B0B0B0, 0x11111111});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xB1313131);
}

TEST_F(GcnTest, and_or_b32_6) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOmod(Omod::Mul2).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0x40404040, 0xB0B0B0B0, 0x11111111});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x11111111);
}

TEST_F(GcnTest, and_or_b32_7) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOmod(Omod::Div2).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0xB0B0B0B0, 0x77777777, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x70707070);
}

TEST_F(GcnTest, and_or_b32_8) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetAbs({1,1,0}).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0xB0B0B0B0, 0x11111111, 0x11111111});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x11111111);
}

TEST_F(GcnTest, mad_f32_rounds_multiply_before_add) {
    auto runner = gcn_test::Runner::instance().value();

    // This triple distinguishes GCN V_MAD_F32 from a fused multiply-add. The rounded
    // float32 product exactly cancels src2, while an FMA retains a non-zero residual.
    constexpr float a = 1.00000011920928955078125f;
    constexpr float b = 1.00000011920928955078125f;
    constexpr float c = -1.0000002384185791015625f;
    ASSERT_NE(std::fma(a, b, c), 0.0f);

    const auto inst =
        VOP3A(OpcodeVOP3::V_MAD_F32, VOperand8::V0, SOperand9::V0, SOperand9::V1,
              SOperand9::V2)
            .Get();
    const auto spirv = TranslateToSpirv(inst);
    const auto result = runner->run<float>(spirv, std::array{a, b, c});

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0.0f);
}

TEST_F(GcnTest, mad_mix_f32_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_MAD_MIX_F32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOpSelHi({0}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<float>(spirv, std::array{2.0f, 3.0f, 4.0f});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 10.0f);
}

TEST_F(GcnTest, mad_mix_f32_2) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_MAD_MIX_F32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOpSelHi({1,1,0}).SetOpSel({1,0,0}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<float>(spirv, std::array<u32,3>{
        std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(4.0f)}
    );

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 26.0f);
}

TEST_F(GcnTest, mad_mixlo_f16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_MAD_MIXLO_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOpSelHi({1,1,0}).SetOpSel({1,0,0}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<F16x2>(spirv, std::array<u32,3>{
        std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(4.0f)}
    );

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(26.0f), half(0.5f)}));
}

TEST_F(GcnTest, mad_mixhi_f16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_MAD_MIXHI_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOpSelHi({1,1,0}).SetOpSel({1,0,0}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<F16x2>(spirv, std::array<u32,3>{
        std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(4.0f)}
    );

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(44.0f), half(26.0f)}));
}

TEST_F(GcnTest, lshrrev_b16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_LSHRREV_B16, VOperand8::V0, SOperand9::V0, SOperand9::V1).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,2>{0xFFFFFFF2, 0x88881000});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xFFFF0400);
}

TEST_F(GcnTest, lshrrev_b16_2) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_LSHRREV_B16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetOpSel({0,0,0,1}).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,2>{0xFFFFFFF2, 0x88881000});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x0400FFF2);
}

TEST_F(GcnTest, lshrrev_b16_3) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_LSHRREV_B16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetOpSel({0,1,0,0}).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,2>{0xFFFFFFF2, 0x88881000});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xFFFF2222);
}

TEST_F(GcnTest, lshlrev_b16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_LSHLREV_B16, VOperand8::V0, SOperand9::V0, SOperand9::V1).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,2>{0xFFFFFFF3, 0x88888888});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xFFFF4440);
}

TEST_F(GcnTest, ashrrev_i16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ASHRREV_I16, VOperand8::V0, SOperand9::V0, SOperand9::V1).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,2>{0x1234FFF3, 0x88888888});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x1234F111);
}

TEST_F(GcnTest, pk_add_f16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f), half(2.0f)}, F16x2{half(3.0f), half(4.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(4.0f), half(6.0f)}));
}

TEST_F(GcnTest, pk_add_f16_2) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::Const0, SOperand9::ConstInv2Pi).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<u32>(spirv, 0U);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x00003118);
}

TEST_F(GcnTest, pk_add_f16_3) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::Const0, SOperand9::ConstInv2Pi).SetOpSel({0,1,1}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<u32>(spirv, 0U);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0);
}

TEST_F(GcnTest, pk_add_f16_4) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::Const0p5, SOperand9::Const0p5).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<u32>(spirv, 0U);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x3C00);
}

TEST_F(GcnTest, pk_add_f16_5) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::Const0, SOperand9::ConstInv2Pi).SetOpSelHi({0,0,0}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<u32>(spirv, 0U);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x31183118);
}

TEST_F(GcnTest, pk_add_f16_neg_lo) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetNeg({1,1,0}).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f), half(2.0f)}, F16x2{half(3.0f), half(4.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(-4.0f), half(6.0f)}));
}

TEST_F(GcnTest, pk_add_f16_neg_hi) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetNegHi({1,1,0}).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f), half(2.0f)}, F16x2{half(3.0f), half(4.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(4.0f), half(-6.0f)}));
}

TEST_F(GcnTest, pk_add_f16_op_sel_reversed) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetOpSel({1,1,1}).SetOpSelHi({0,0,0}).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f), half(2.0f)}, F16x2{half(3.0f), half(4.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(6.0f), half(4.0f)}));
}
