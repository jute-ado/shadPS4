// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/recompiler.h"

namespace Shader::Optimization {
namespace {

TEST(LowerBufferFormatToRaw, InvalidFormattedStoreBecomesANoOp) {
    Info info{};
    info.flattened_ud_buf.resize(4);
    info.buffers.push_back(BufferResource{
        .sharp_idx = 0,
        .used_types = IR::Type::F32,
        .buffer_type = BufferType::Guest,
        .is_written = true,
        .is_formatted = true,
    });

    Pools pools{};
    IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    IR::IREmitter ir{*block};
    const auto value = ir.CompositeConstruct(ir.Imm32(1.0F), ir.Imm32(2.0F), ir.Imm32(3.0F),
                                             ir.Imm32(4.0F));
    ir.StoreBufferFormat(ir.Imm32(0U), ir.Imm32(0U), value, {});

    EXPECT_NO_THROW(LowerBufferFormatToRaw(program));
    EXPECT_EQ(block->Instructions().back().GetOpcode(), IR::Opcode::Void);
}

} // namespace
} // namespace Shader::Optimization
