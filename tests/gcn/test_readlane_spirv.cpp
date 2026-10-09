// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "spirv/unified1/spirv.hpp"

namespace {
using namespace Shader;

enum class LaneSource { ConstantZero, InvocationXor, FirstActive };

std::vector<u32> EmitLaneRead(LaneSource source) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    Pools pools{};
    IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    auto& block_node = program.syntax_list.emplace_back();
    block_node.type = IR::AbstractSyntaxNode::Type::Block;
    block_node.data.block = block;
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    IR::IREmitter ir{*block};
    ir.Prologue();
    const auto lane = ir.LaneId();
    const auto value = ir.IAdd(lane, ir.Imm32(100U));
    if (source == LaneSource::FirstActive) {
        ir.Reference(ir.ReadFirstLane(IR::U32{value}));
    } else {
        const IR::U32 target{source == LaneSource::InvocationXor
                                 ? IR::Value{ir.BitwiseXor(lane, ir.Imm32(16U))}
                                 : IR::Value{ir.Imm32(0U)}};
        ir.Reference(ir.ReadLane(IR::U32{value}, target));
    }
    ir.Epilogue();
    Profile profile{};
    profile.supported_spirv = 0x00010600U;
    profile.subgroup_size = 32U;
    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Compute);
    runtime.cs_info.workgroup_size = {32U, 1U, 1U};
    Optimization::CollectShaderInfoPass(program, profile);
    Backend::Bindings bindings{};
    auto spirv = Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings);
    EXPECT_FALSE(info.translation_failed);
    return spirv;
}

size_t CountInstruction(const std::vector<u32>& module, spv::Op operation) {
    size_t count{};
    for (size_t offset = 5; offset < module.size();) {
        const u32 width = module[offset] >> 16U;
        EXPECT_GT(width, 0U);
        if (width == 0U || width > module.size() - offset) {
            return 0U;
        }
        count += (module[offset] & 0xFFFFU) == static_cast<u32>(operation);
        offset += width;
    }
    return count;
}

bool HasCapability(const std::vector<u32>& module, spv::Capability capability) {
    for (size_t offset = 5; offset < module.size();) {
        const u32 width = module[offset] >> 16U;
        if (width == 0U || width > module.size() - offset) {
            return false;
        }
        if ((module[offset] & 0xFFFFU) == static_cast<u32>(spv::Op::OpCapability) &&
            width == 2U && module[offset + 1] == static_cast<u32>(capability)) {
            return true;
        }
        offset += width;
    }
    return false;
}

TEST(ReadLaneSpirv, InvocationVaryingLaneUsesShuffleNotUniformBroadcast) {
    const auto module = EmitLaneRead(LaneSource::InvocationXor);
    EXPECT_EQ(CountInstruction(module, spv::Op::OpGroupNonUniformShuffle), 1U);
    EXPECT_EQ(CountInstruction(module, spv::Op::OpGroupNonUniformBroadcast), 0U);
    EXPECT_TRUE(HasCapability(module, spv::CapabilityGroupNonUniformShuffle));
}

TEST(ReadLaneSpirv, ImmediateLaneRetainsUniformBroadcast) {
    const auto module = EmitLaneRead(LaneSource::ConstantZero);
    EXPECT_EQ(CountInstruction(module, spv::Op::OpGroupNonUniformBroadcast), 1U);
    EXPECT_EQ(CountInstruction(module, spv::Op::OpGroupNonUniformShuffle), 0U);
    EXPECT_FALSE(HasCapability(module, spv::CapabilityGroupNonUniformShuffle));
}

TEST(ReadLaneSpirv, FirstActiveLaneRetainsItsBroadcastSemantics) {
    const auto module = EmitLaneRead(LaneSource::FirstActive);
    EXPECT_EQ(CountInstruction(module, spv::Op::OpGroupNonUniformBroadcastFirst), 1U);
    EXPECT_EQ(CountInstruction(module, spv::Op::OpGroupNonUniformShuffle), 0U);
}
} // namespace
