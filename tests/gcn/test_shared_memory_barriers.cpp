// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include <gtest/gtest.h>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"

namespace {

TEST(SharedMemoryBarrierPass, SubWaveThreadgroupOrdersStraightLineLdsWriteThenRead) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    Shader::IR::IREmitter ir{*block};
    ir.WriteShared(32, ir.Imm32(7), ir.Imm32(0));
    ir.Reference(ir.LoadShared(32, false, ir.Imm32(0)));
    ir.Epilogue();
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.shared_memory_size = 256;
    runtime_info.cs_info.workgroup_size = {16, 1, 1};
    Shader::Profile profile{};
    profile.needs_lds_barriers = true;

    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);

    const auto barrier = std::ranges::find_if(block->Instructions(), [](const auto& inst) {
        return inst.GetOpcode() == Shader::IR::Opcode::Barrier;
    });
    EXPECT_NE(barrier, block->Instructions().end());
}

} // namespace
