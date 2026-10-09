// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include <gtest/gtest.h>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/recompiler.h"

namespace {

AmdGpu::Buffer MakeBufferSharp(AmdGpu::DataFormat data_format,
                               AmdGpu::NumberFormat number_format) {
    AmdGpu::Buffer sharp{};
    sharp.num_records = 1;
    sharp.data_format = static_cast<u32>(data_format);
    sharp.num_format = static_cast<u32>(number_format);
    sharp.dst_sel_x = static_cast<u32>(AmdGpu::CompSwizzle::Red);
    sharp.dst_sel_y = static_cast<u32>(AmdGpu::CompSwizzle::Green);
    sharp.dst_sel_z = static_cast<u32>(AmdGpu::CompSwizzle::Blue);
    sharp.dst_sel_w = static_cast<u32>(AmdGpu::CompSwizzle::Alpha);
    return sharp;
}

void AddFormattedBuffer(Shader::Info& info, const AmdGpu::Buffer& sharp, bool is_written) {
    info.flattened_ud_buf.resize(4);
    std::memcpy(info.flattened_ud_buf.data(), &sharp, sizeof(sharp));
    info.buffers.push_back(Shader::BufferResource{
        .sharp_idx = 0,
        .used_types = Shader::IR::Type::F32,
        .buffer_type = Shader::BufferType::Guest,
        .is_written = is_written,
        .is_formatted = true,
    });
}

bool ContainsOpcode(const Shader::IR::Block& block, Shader::IR::Opcode opcode) {
    for (const Shader::IR::Inst& inst : block.Instructions()) {
        if (inst.GetOpcode() == opcode) {
            return true;
        }
    }
    return false;
}

} // namespace

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

TEST(LowerBufferFormatToRaw, DiscardsStoreWithUnsupportedFloat8x4DescriptorPair) {
    Shader::Info info{};
    AddFormattedBuffer(info,
                       MakeBufferSharp(AmdGpu::DataFormat::Format8_8_8_8,
                                       AmdGpu::NumberFormat::Float),
                       true);

    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    const auto data = ir.CompositeConstruct(ir.Imm32(1.f), ir.Imm32(2.f), ir.Imm32(3.f),
                                            ir.Imm32(4.f));
    ir.StoreBufferFormat(ir.Imm32(0U), ir.Imm32(0U), data, {});

    Shader::Optimization::LowerBufferFormatToRaw(program);

    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::StoreBufferFormatF32));
    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::PackUnorm4x8));
    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::PackSnorm4x8));
    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::PackUint4x8));
    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::PackSint4x8));
    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::StoreBufferU32));
}

TEST(LowerBufferFormatToRaw, DiscardsStoreWithUnsupportedFloat8x2DescriptorPair) {
    Shader::Info info{};
    AddFormattedBuffer(
        info, MakeBufferSharp(AmdGpu::DataFormat::Format8_8, AmdGpu::NumberFormat::Float), true);

    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    const auto data = ir.CompositeConstruct(ir.Imm32(1.f), ir.Imm32(2.f), ir.Imm32(3.f),
                                            ir.Imm32(4.f));
    ir.StoreBufferFormat(ir.Imm32(0U), ir.Imm32(0U), data, {});

    Shader::Optimization::LowerBufferFormatToRaw(program);

    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::StoreBufferFormatF32));
    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::StoreBufferU16));
}

TEST(LowerBufferFormatToRaw, DiscardsStoreWithUnsupportedFloat8x4InstructionPair) {
    Shader::Info info{};
    AddFormattedBuffer(
        info, MakeBufferSharp(AmdGpu::DataFormat::Format32, AmdGpu::NumberFormat::Float), true);

    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::BufferInstInfo inst_info{};
    inst_info.inst_data_fmt.Assign(AmdGpu::DataFormat::Format8_8_8_8);
    inst_info.inst_num_fmt.Assign(AmdGpu::NumberFormat::Float);
    Shader::IR::IREmitter ir{*block};
    const auto data = ir.CompositeConstruct(ir.Imm32(1.f), ir.Imm32(2.f), ir.Imm32(3.f),
                                            ir.Imm32(4.f));
    ir.StoreBufferFormat(ir.Imm32(0U), ir.Imm32(0U), data, inst_info);

    Shader::Optimization::LowerBufferFormatToRaw(program);

    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::StoreBufferFormatF32));
    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::StoreBufferU32));
}

TEST(LowerBufferFormatToRaw, ZerosLoadWithUnsupportedFloat8x4DescriptorPair) {
    Shader::Info info{};
    AddFormattedBuffer(info,
                       MakeBufferSharp(AmdGpu::DataFormat::Format8_8_8_8,
                                       AmdGpu::NumberFormat::Float),
                       false);

    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    ir.Reference(ir.LoadBufferFormat(ir.Imm32(0U), ir.Imm32(0U), {}));

    Shader::Optimization::LowerBufferFormatToRaw(program);

    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::LoadBufferFormatF32));
    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::UnpackUnorm4x8));
    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::UnpackSnorm4x8));
    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::UnpackUint4x8));
    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::UnpackSint4x8));
    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::LoadBufferU32));
}

TEST(LowerBufferFormatToRaw, LowersSupportedUnorm8x4Store) {
    Shader::Info info{};
    AddFormattedBuffer(info,
                       MakeBufferSharp(AmdGpu::DataFormat::Format8_8_8_8,
                                       AmdGpu::NumberFormat::Unorm),
                       true);

    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    const auto data = ir.CompositeConstruct(ir.Imm32(1.f), ir.Imm32(2.f), ir.Imm32(3.f),
                                            ir.Imm32(4.f));
    ir.StoreBufferFormat(ir.Imm32(0U), ir.Imm32(0U), data, {});

    Shader::Optimization::LowerBufferFormatToRaw(program);

    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::StoreBufferFormatF32));
    EXPECT_TRUE(ContainsOpcode(*block, Shader::IR::Opcode::PackUnorm4x8));
    EXPECT_TRUE(ContainsOpcode(*block, Shader::IR::Opcode::StoreBufferU32));
}

TEST(LowerBufferFormatToRaw, LowersSupportedFloat16x4Store) {
    Shader::Info info{};
    AddFormattedBuffer(info,
                       MakeBufferSharp(AmdGpu::DataFormat::Format16_16_16_16,
                                       AmdGpu::NumberFormat::Float),
                       true);

    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    const auto data = ir.CompositeConstruct(ir.Imm32(1.f), ir.Imm32(2.f), ir.Imm32(3.f),
                                            ir.Imm32(4.f));
    ir.StoreBufferFormat(ir.Imm32(0U), ir.Imm32(0U), data, {});

    Shader::Optimization::LowerBufferFormatToRaw(program);

    EXPECT_FALSE(ContainsOpcode(*block, Shader::IR::Opcode::StoreBufferFormatF32));
    EXPECT_TRUE(ContainsOpcode(*block, Shader::IR::Opcode::PackHalf2x16));
    EXPECT_TRUE(ContainsOpcode(*block, Shader::IR::Opcode::StoreBufferU32x2));
}
