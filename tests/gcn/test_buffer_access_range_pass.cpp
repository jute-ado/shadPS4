// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include <gtest/gtest.h>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"

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

TEST(LowerBufferFormatToRaw, DiscardsFormattedStoreToInvalidNullDescriptor) {
    Shader::Info info{};
    info.flattened_ud_buf.resize(4);
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
    const auto value =
        ir.CompositeConstruct(ir.Imm32(1.0f), ir.Imm32(2.0f), ir.Imm32(3.0f), ir.Imm32(4.0f));
    ir.StoreBufferFormat(ir.Imm32(0U), ir.Imm32(0U), value, {});

    Shader::Optimization::LowerBufferFormatToRaw(program);

    for (const auto& inst : block->Instructions()) {
        EXPECT_NE(inst.GetOpcode(), Shader::IR::Opcode::StoreBufferFormatF32);
    }
}

TEST(LowerBufferFormatToRaw, DiscardsFormattedStoreWithUnsupportedFormatPair) {
    Shader::Info info{};
    info.flattened_ud_buf.resize(4);

    AmdGpu::Buffer sharp{};
    sharp.base_address = 0x10000;
    sharp.stride = 4;
    sharp.num_records = 16;
    sharp.data_format = static_cast<u32>(AmdGpu::DataFormat::Format2_10_10_10);
    sharp.num_format = static_cast<u32>(AmdGpu::NumberFormat::Float);
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
    const auto value =
        ir.CompositeConstruct(ir.Imm32(1.0f), ir.Imm32(2.0f), ir.Imm32(3.0f), ir.Imm32(4.0f));
    ir.StoreBufferFormat(ir.Imm32(0U), ir.Imm32(0U), value, {});

    Shader::Optimization::LowerBufferFormatToRaw(program);

    for (const auto& inst : block->Instructions()) {
        EXPECT_NE(inst.GetOpcode(), Shader::IR::Opcode::StoreBufferFormatF32);
        EXPECT_NE(inst.GetOpcode(), Shader::IR::Opcode::StoreBufferU32);
    }
}

TEST(LowerBufferFormatToRaw, ZerosFormattedLoadWithUnsupportedFormatPair) {
    Shader::Info info{};
    info.flattened_ud_buf.resize(4);

    AmdGpu::Buffer sharp{};
    sharp.base_address = 0x10000;
    sharp.stride = 4;
    sharp.num_records = 16;
    sharp.data_format = static_cast<u32>(AmdGpu::DataFormat::Format2_10_10_10);
    sharp.num_format = static_cast<u32>(AmdGpu::NumberFormat::Float);
    std::memcpy(info.flattened_ud_buf.data(), &sharp, sizeof(sharp));

    info.buffers.push_back(Shader::BufferResource{
        .sharp_idx = 0,
        .used_types = Shader::IR::Type::F32,
        .buffer_type = Shader::BufferType::Guest,
        .is_formatted = true,
    });

    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    static_cast<void>(ir.LoadBufferFormat(ir.Imm32(0U), ir.Imm32(0U), {}));

    Shader::Optimization::LowerBufferFormatToRaw(program);

    for (const auto& inst : block->Instructions()) {
        EXPECT_NE(inst.GetOpcode(), Shader::IR::Opcode::LoadBufferFormatF32);
        EXPECT_NE(inst.GetOpcode(), Shader::IR::Opcode::LoadBufferU32);
    }
}

TEST(BufferSpecialization, AddThreadIdParticipatesInPermutationIdentity) {
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    info.flattened_ud_buf.resize(4);
    info.buffers.push_back(Shader::BufferResource{
        .sharp_idx = 0,
        .used_types = Shader::IR::Type::U32,
        .buffer_type = Shader::BufferType::Guest,
    });

    AmdGpu::Buffer sharp{};
    sharp.base_address = 0x10000;
    sharp.stride = 4;
    sharp.num_records = 16;
    std::memcpy(info.flattened_ud_buf.data(), &sharp, sizeof(sharp));

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Compute);
    Shader::Profile profile{};
    const Shader::StageSpecialization without_add_tid{info, runtime_info, profile, {}};

    sharp.add_tid_enable = 1;
    std::memcpy(info.flattened_ud_buf.data(), &sharp, sizeof(sharp));
    const Shader::StageSpecialization with_add_tid{info, runtime_info, profile, {}};

    EXPECT_FALSE(without_add_tid == with_add_tid);
    EXPECT_FALSE(with_add_tid == without_add_tid);

    sharp.data_format = 1;
    std::memcpy(info.flattened_ud_buf.data(), &sharp, sizeof(sharp));
    const Shader::StageSpecialization with_low_extended_stride{info, runtime_info, profile, {}};
    sharp.data_format = 2;
    std::memcpy(info.flattened_ud_buf.data(), &sharp, sizeof(sharp));
    const Shader::StageSpecialization with_high_extended_stride{info, runtime_info, profile, {}};

    EXPECT_FALSE(with_low_extended_stride == with_high_extended_stride);
    EXPECT_FALSE(with_high_extended_stride == with_low_extended_stride);
}

TEST(BufferDescriptor, AddThreadIdExtendsStrideOnlyForNonFormatMubuf) {
    AmdGpu::Buffer sharp{};
    sharp.stride = 0x1234;
    sharp.data_format = 0xA;
    sharp.add_tid_enable = 1;

    EXPECT_EQ(sharp.GetStride(true, false), 0x29234U);  // Raw MUBUF.
    EXPECT_EQ(sharp.GetStride(true, true), 0x1234U);    // Formatted MUBUF.
    EXPECT_EQ(sharp.GetStride(false, true), 0x1234U);   // MTBUF.
}
