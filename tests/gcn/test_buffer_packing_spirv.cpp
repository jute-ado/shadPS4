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

u32 CountOpcode(const std::vector<u32>& spirv, const spv::Op opcode) {
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

std::vector<u32> EmitSigned10_11_11PackShader() {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;

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
    const auto values = ir.CompositeConstruct(ir.BitCast<IR::F32>(ir.LaneId()),
                                              ir.BitCast<IR::F32>(ir.Imm32(2)),
                                              ir.BitCast<IR::F32>(ir.Imm32(3)));
    ir.Reference(ir.Pack10_11_11(AmdGpu::NumberFormat::Sint, values));
    ir.Reference(ir.Unpack10_11_11(AmdGpu::NumberFormat::Sint, ir.LaneId()));
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    Optimization::CollectShaderInfoPass(program, profile);
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(info.stage);
    runtime_info.cs_info.workgroup_size = {32, 1, 1};
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

std::vector<u32> EmitVectorPairConcatenationShader() {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;

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
    const auto x = ir.BitCast<IR::F32>(ir.LaneId());
    const auto first = ir.CompositeConstruct(x, ir.Imm32(1.f));
    const auto second = ir.CompositeConstruct(ir.Imm32(2.f), ir.Imm32(3.f));
    const auto combined = ir.CompositeConstruct(first, second);
    ir.Reference(ir.CompositeExtract(combined, 2));
    ir.Reference(ir.CompositeExtract(combined, 3));
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    Optimization::CollectShaderInfoPass(program, profile);
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(info.stage);
    runtime_info.cs_info.workgroup_size = {32, 1, 1};
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

TEST(BufferPackingSpirv, EmitsSigned10_11_11Packing) {
    const auto spirv = EmitSigned10_11_11PackShader();

    EXPECT_GE(CountOpcode(spirv, spv::Op::OpBitFieldInsert), 2u);
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpBitFieldSExtract), 3u);
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpBitcast), 1u);
}

TEST(BufferPackingSpirv, ConcatenatesVectorPairsWithAValidVectorShuffle) {
    const auto spirv = EmitVectorPairConcatenationShader();

    EXPECT_GE(CountOpcode(spirv, spv::Op::OpVectorShuffle), 1u);
}

} // namespace
