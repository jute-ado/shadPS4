// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/recompiler.h"

TEST(SsaRewritePass, ReusesUndefinitionsAcrossDisconnectedBlocks) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const first = pools.block_pool.Create(pools.inst_pool);
    auto* const second = pools.block_pool.Create(pools.inst_pool);
    program.blocks = {first, second};
    program.post_order_blocks = {second, first};

    Shader::IR::IREmitter first_ir{*first};
    first_ir.Reference(first_ir.GetScalarReg(Shader::IR::ScalarReg::S0));
    Shader::IR::IREmitter second_ir{*second};
    second_ir.Reference(second_ir.GetScalarReg(Shader::IR::ScalarReg::S0));

    Shader::Optimization::SsaRewritePass(program);

    const Shader::IR::Value first_undef = first->back().Arg(0).Resolve();
    const Shader::IR::Value second_undef = second->back().Arg(0).Resolve();
    ASSERT_NE(first_undef.TryInstRecursive(), nullptr);
    EXPECT_EQ(first_undef.TryInstRecursive()->GetOpcode(), Shader::IR::Opcode::UndefU32);
    EXPECT_EQ(first_undef, second_undef);
    EXPECT_EQ(first_undef.TryInstRecursive()->GetParent(), first);
}
