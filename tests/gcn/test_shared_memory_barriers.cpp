// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"

namespace {

size_t CountInsertedBarriers(u32 workgroup_size, u32 shared_memory_size = 256,
                             bool needs_lds_barriers = true) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    Shader::IR::Program program{info};
    Shader::Pools pools{};

    Shader::IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter ir{*block};
    ir.WriteShared(32, ir.Imm32(42U), ir.Imm32(0U));
    ir.Epilogue();

    auto& block_node = program.syntax_list.emplace_back();
    block_node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
    block_node.data.block = block;
    program.syntax_list.emplace_back().type = Shader::IR::AbstractSyntaxNode::Type::Return;

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    runtime_info.cs_info.shared_memory_size = shared_memory_size;
    runtime_info.cs_info.workgroup_size = {workgroup_size, 1, 1};

    Shader::Profile profile{};
    profile.needs_lds_barriers = needs_lds_barriers;
    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);

    return std::ranges::count_if(block->Instructions(), [](const Shader::IR::Inst& inst) {
        return inst.GetOpcode() == Shader::IR::Opcode::Barrier;
    });
}

TEST(SharedMemoryBarrierPass, SynchronizesPartialGuestWaveWorkgroups) {
    EXPECT_EQ(CountInsertedBarriers(32), 1);
}

TEST(SharedMemoryBarrierPass, SynchronizesFullGuestWaveWorkgroups) {
    EXPECT_EQ(CountInsertedBarriers(64), 1);
}

TEST(SharedMemoryBarrierPass, LeavesMultiWaveWorkgroupsUnchanged) {
    EXPECT_EQ(CountInsertedBarriers(128), 0);
}

TEST(SharedMemoryBarrierPass, SkipsSynchronizationWhenHostDoesNotNeedIt) {
    EXPECT_EQ(CountInsertedBarriers(32, 256, false), 0);
}

TEST(SharedMemoryBarrierPass, SkipsWorkgroupsWithoutSharedMemory) {
    EXPECT_EQ(CountInsertedBarriers(32, 0), 0);
}

} // namespace
