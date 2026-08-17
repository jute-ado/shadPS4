// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include <gtest/gtest.h>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/recompiler.h"

TEST(BufferAccessRangePass, TracksFormattedAccessAfterItIsLoweredToRaw) {
    Shader::Info info{};
    info.flattened_ud_buf.resize(4);

    AmdGpu::Buffer sharp{};
    sharp.stride = 4;
    sharp.num_records = 0x3fffffffU;
    std::memcpy(info.flattened_ud_buf.data(), &sharp, sizeof(sharp));

    info.buffers.push_back(Shader::BufferResource{
        .sharp_idx = 0,
        .used_types = Shader::IR::Type::U32,
        .buffer_type = Shader::BufferType::Guest,
        .is_formatted = true,
    });

    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    static_cast<void>(ir.LoadBufferU32(4, ir.Imm32(0U), ir.Imm32(12U), {}));

    Shader::Optimization::BufferAccessRangePass(program);

    ASSERT_EQ(info.buffers.size(), 1);
    EXPECT_TRUE(info.buffers[0].access_range.IsBounded());
    EXPECT_EQ(info.buffers[0].access_range.UpperBound(), 64);
    EXPECT_EQ(info.buffers[0].GetBindingSize(sharp), 64);
}

