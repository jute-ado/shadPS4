// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "common/object_pool.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/recompiler.h"
#include "spirv/unified1/spirv.hpp"

namespace {

using namespace Shader;

std::vector<u32> EmitDynamicFloatClassShader() {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
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
    const IR::U32 dynamic_value = ir.LaneId();
    ir.Reference(ir.FPCmpClass32(ir.BitCast<IR::F32>(dynamic_value), dynamic_value));
    ir.Epilogue();

    Optimization::ConstantPropagationPass(program.blocks);

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(info.stage);
    runtime_info.cs_info.workgroup_size = {32, 1, 1};
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

u32 CountOpcode(const std::vector<u32>& spirv, spv::Op opcode) {
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

TEST(FloatClassSpirv, LowersDynamicClassMaskWithoutAssumingAnImmediate) {
    const auto spirv = EmitDynamicFloatClassShader();

    EXPECT_GE(CountOpcode(spirv, spv::Op::OpBitcast), 1u);
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpBitwiseAnd), 2u);
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpSelect), 5u);
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpINotEqual), 1u);
}

} // namespace
