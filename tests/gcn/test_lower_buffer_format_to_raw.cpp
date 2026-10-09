// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include <gtest/gtest.h>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/recompiler.h"

TEST(LowerBufferFormatToRaw, DiscardsStoreWithInvalidDescriptorFormat) {
    Shader::Info info{};
    info.flattened_ud_buf.resize(4);

    AmdGpu::Buffer sharp{};
    sharp.num_records = 1;
    sharp.data_format = static_cast<u32>(AmdGpu::DataFormat::FormatInvalid);
    std::memcpy(info.flattened_ud_buf.data(), &sharp, sizeof(sharp));

    info.buffers.push_back(Shader::BufferResource{
        .sharp_idx = 0,
        .used_types = Shader::IR::Type::F32,
        .buffer_type = Shader::BufferType::Guest,
        .is_written = true,
        .is_formatted = true,
    });

    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    const auto data = ir.CompositeConstruct(ir.Imm32(1.f), ir.Imm32(2.f), ir.Imm32(3.f),
                                            ir.Imm32(4.f));
    ir.StoreBufferFormat(ir.Imm32(0U), ir.Imm32(0U), data, {});

    Shader::Optimization::LowerBufferFormatToRaw(program);

    for (const Shader::IR::Inst& inst : block->Instructions()) {
        EXPECT_NE(inst.GetOpcode(), Shader::IR::Opcode::StoreBufferFormatF32);
    }
}
