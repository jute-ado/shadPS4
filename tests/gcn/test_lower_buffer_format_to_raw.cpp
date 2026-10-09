// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>

#include <gtest/gtest.h>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/recompiler.h"

namespace {

Shader::Info MakeFormattedBufferInfo(AmdGpu::DataFormat data_format,
                                     AmdGpu::NumberFormat number_format, bool is_written = false) {
    Shader::Info info{};
    info.flattened_ud_buf.resize(4);

    AmdGpu::Buffer sharp{};
    sharp.num_records = 1;
    sharp.data_format = static_cast<u32>(data_format);
    sharp.num_format = static_cast<u32>(number_format);
    sharp.dst_sel_x = static_cast<u32>(AmdGpu::CompSwizzle::Red);
    sharp.dst_sel_y = static_cast<u32>(AmdGpu::CompSwizzle::Green);
    sharp.dst_sel_z = static_cast<u32>(AmdGpu::CompSwizzle::Blue);
    sharp.dst_sel_w = static_cast<u32>(AmdGpu::CompSwizzle::Alpha);
    std::memcpy(info.flattened_ud_buf.data(), &sharp, sizeof(sharp));

    info.buffers.push_back(Shader::BufferResource{
        .sharp_idx = 0,
        .used_types = Shader::IR::Type::F32,
        .buffer_type = Shader::BufferType::Guest,
        .is_written = is_written,
        .is_formatted = true,
    });
    return info;
}

u32 CountOpcode(const Shader::IR::Block& block, Shader::IR::Opcode opcode) {
    u32 count{};
    for (const Shader::IR::Inst& inst : block.Instructions()) {
        count += inst.GetOpcode() == opcode;
    }
    return count;
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

TEST(LowerBufferFormatToRaw, InvalidDescriptorOverridesTypedInstructionFormat) {
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
    Shader::IR::BufferInstInfo flags{};
    flags.typed.Assign(true);
    flags.inst_data_fmt.Assign(AmdGpu::DataFormat::Format32);
    flags.inst_num_fmt.Assign(AmdGpu::NumberFormat::Float);
    ir.StoreBufferFormat(ir.Imm32(0U), ir.Imm32(0U), data, flags);

    Shader::Optimization::LowerBufferFormatToRaw(program);

    for (const Shader::IR::Inst& inst : block->Instructions()) {
        EXPECT_NE(inst.GetOpcode(), Shader::IR::Opcode::StoreBufferFormatF32);
        EXPECT_NE(inst.GetOpcode(), Shader::IR::Opcode::StoreBufferF32);
    }
}

TEST(LowerBufferFormatToRaw, LowersValidFormattedStoreToRawAccess) {
    Shader::Info info{};
    info.flattened_ud_buf.resize(4);

    AmdGpu::Buffer sharp{};
    sharp.num_records = 1;
    sharp.data_format = static_cast<u32>(AmdGpu::DataFormat::Format32);
    sharp.num_format = static_cast<u32>(AmdGpu::NumberFormat::Float);
    sharp.dst_sel_x = static_cast<u32>(AmdGpu::CompSwizzle::Red);
    sharp.dst_sel_y = static_cast<u32>(AmdGpu::CompSwizzle::Green);
    sharp.dst_sel_z = static_cast<u32>(AmdGpu::CompSwizzle::Blue);
    sharp.dst_sel_w = static_cast<u32>(AmdGpu::CompSwizzle::Alpha);
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

    bool found_raw_store = false;
    for (const Shader::IR::Inst& inst : block->Instructions()) {
        EXPECT_NE(inst.GetOpcode(), Shader::IR::Opcode::StoreBufferFormatF32);
        found_raw_store |= inst.GetOpcode() == Shader::IR::Opcode::StoreBufferF32;
    }
    EXPECT_TRUE(found_raw_store);
}

TEST(LowerBufferFormatToRaw, Converts32BitSnormNzLoadsWithoutIntegerOverflow) {
    auto info =
        MakeFormattedBufferInfo(AmdGpu::DataFormat::Format32_32_32, AmdGpu::NumberFormat::SnormNz);
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    (void)ir.LoadBufferFormat(ir.Imm32(0U), ir.Imm32(0U), {});

    Shader::Optimization::LowerBufferFormatToRaw(program);

    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::LoadBufferFormatF32), 0U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::LoadBufferF32x3), 1U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::ConvertF32S32), 3U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::FPAdd32), 3U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::FPDiv32), 3U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::IAdd32), 0U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::IMul32), 0U);

    for (const Shader::IR::Inst& inst : block->Instructions()) {
        if (inst.GetOpcode() != Shader::IR::Opcode::FPDiv32) {
            continue;
        }
        ASSERT_TRUE(inst.Arg(1).IsImmediate());
        EXPECT_EQ(inst.Arg(1).F32(), 2147483648.0f);

        const auto* numerator = inst.Arg(0).Inst();
        ASSERT_EQ(numerator->GetOpcode(), Shader::IR::Opcode::FPAdd32);
        ASSERT_TRUE(numerator->Arg(1).IsImmediate());
        EXPECT_EQ(numerator->Arg(1).F32(), 0.5f);
    }
}

TEST(LowerBufferFormatToRaw, Converts32BitSnormNzStoresWithSaturatedEndpoints) {
    auto info = MakeFormattedBufferInfo(AmdGpu::DataFormat::Format32_32_32,
                                        AmdGpu::NumberFormat::SnormNz, true);
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    const auto data = ir.CompositeConstruct(ir.Imm32(-1.f), ir.Imm32(0.f), ir.Imm32(1.f),
                                            ir.Imm32(0.f));
    ir.StoreBufferFormat(ir.Imm32(0U), ir.Imm32(0U), data, {});

    Shader::Optimization::LowerBufferFormatToRaw(program);

    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::StoreBufferFormatF32), 0U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::StoreBufferF32x3), 1U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::ConvertS32F32), 3U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::FPOrdGreaterThanEqual32), 3U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::FPOrdLessThanEqual32), 3U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::SelectU32), 6U);

    // The conversion executes before the final integer select. Its floating input must itself
    // exclude +2^31; merely selecting INT32_MAX afterwards does not avoid SPIR-V undefined behavior.
    for (const Shader::IR::Inst& inst : block->Instructions()) {
        if (inst.GetOpcode() != Shader::IR::Opcode::ConvertS32F32) {
            continue;
        }
        const auto* safe_input = inst.Arg(0).InstRecursive();
        ASSERT_NE(safe_input, nullptr);
        ASSERT_EQ(safe_input->GetOpcode(), Shader::IR::Opcode::FPClamp32);
        EXPECT_EQ(safe_input->Arg(1).F32(), -2147483648.f);
        EXPECT_EQ(safe_input->Arg(2).F32(), 2147483520.f);
    }

    bool found_positive_endpoint = false;
    bool found_negative_endpoint = false;
    for (const Shader::IR::Inst& inst : block->Instructions()) {
        for (size_t arg_index = 0; arg_index < inst.NumArgs(); ++arg_index) {
            const Shader::IR::Value& arg = inst.Arg(arg_index);
            if (!arg.IsImmediate() || arg.Type() != Shader::IR::Type::U32) {
                continue;
            }
            found_positive_endpoint |= arg.U32() == 0x7FFFFFFFU;
            found_negative_endpoint |= arg.U32() == 0x80000000U;
        }
    }
    EXPECT_TRUE(found_positive_endpoint);
    EXPECT_TRUE(found_negative_endpoint);
}

TEST(LowerBufferFormatToRaw, RejectsPacked2_10_10_10SnormNzLoadSnapshot) {
    auto info = MakeFormattedBufferInfo(AmdGpu::DataFormat::Format2_10_10_10,
                                        AmdGpu::NumberFormat::SnormNz);
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    (void)ir.LoadBufferFormat(ir.Imm32(0U), ir.Imm32(0U), {});

    Shader::Optimization::LowerBufferFormatToRaw(program);

    EXPECT_TRUE(info.translation_failed);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::LoadBufferFormatF32), 0U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::LoadBufferU32), 0U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::UnpackSint2_10_10_10), 0U);
}

TEST(LowerBufferFormatToRaw, RejectsPacked2_10_10_10SnormNzStoreSnapshot) {
    auto info = MakeFormattedBufferInfo(AmdGpu::DataFormat::Format2_10_10_10,
                                        AmdGpu::NumberFormat::SnormNz, true);
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    const auto data = ir.CompositeConstruct(ir.Imm32(-1.f), ir.Imm32(0.f), ir.Imm32(1.f),
                                            ir.Imm32(0.f));
    ir.StoreBufferFormat(ir.Imm32(0U), ir.Imm32(0U), data, {});

    Shader::Optimization::LowerBufferFormatToRaw(program);

    EXPECT_TRUE(info.translation_failed);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::StoreBufferFormatF32), 0U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::StoreBufferU32), 0U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::PackSint2_10_10_10), 0U);
}

TEST(LowerBufferFormatToRaw, RejectsUnsupportedPackedFloatShaderSnapshot) {
    auto info = MakeFormattedBufferInfo(AmdGpu::DataFormat::Format2_10_10_10,
                                        AmdGpu::NumberFormat::Float, true);
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    ir.Reference(ir.LoadBufferFormat(ir.Imm32(0U), ir.Imm32(0U), {}));
    const auto data = ir.CompositeConstruct(ir.Imm32(1.f), ir.Imm32(2.f), ir.Imm32(3.f),
                                            ir.Imm32(4.f));
    ir.StoreBufferFormat(ir.Imm32(0U), ir.Imm32(0U), data, {});

    Shader::Optimization::LowerBufferFormatToRaw(program);

    EXPECT_TRUE(info.translation_failed);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::LoadBufferFormatF32), 0U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::StoreBufferFormatF32), 0U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::LoadBufferU32), 0U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::StoreBufferU32), 0U);
}

TEST(LowerBufferFormatToRaw, Signed10_11_11PreservesPackedIntegerPayloads) {
    Shader::Info info{};
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    const auto signed_components = ir.CompositeConstruct(
        ir.BitCast<Shader::IR::F32>(ir.Imm32(-1024)),
        ir.BitCast<Shader::IR::F32>(ir.Imm32(1023)),
        ir.BitCast<Shader::IR::F32>(ir.Imm32(-512)));
    ir.Reference(ir.Pack10_11_11(AmdGpu::NumberFormat::Sint, signed_components));

    const auto unpacked =
        ir.Unpack10_11_11(AmdGpu::NumberFormat::Sint, ir.Imm32(0x801FFC00U));
    for (u32 component = 0; component < 3; ++component) {
        ir.Reference(ir.BitCast<Shader::IR::U32>(
            Shader::IR::F32{ir.CompositeExtract(unpacked, component)}));
    }

    Shader::Optimization::ConstantPropagationPass(program.blocks);

    std::array<u32, 4> referenced_values{};
    u32 reference_index{};
    for (const Shader::IR::Inst& inst : block->Instructions()) {
        if (inst.GetOpcode() != Shader::IR::Opcode::Reference) {
            continue;
        }
        ASSERT_LT(reference_index, referenced_values.size());
        ASSERT_TRUE(inst.Arg(0).IsImmediate());
        referenced_values[reference_index++] = inst.Arg(0).U32();
    }
    ASSERT_EQ(reference_index, referenced_values.size());
    EXPECT_EQ(referenced_values,
              (std::array{0x801FFC00U, static_cast<u32>(-1024), 1023U,
                          static_cast<u32>(-512)}));
}

TEST(LowerBufferFormatToRaw, UnpacksSigned10_11_11DescriptorLoads) {
    auto info =
        MakeFormattedBufferInfo(AmdGpu::DataFormat::Format10_11_11, AmdGpu::NumberFormat::Sint);
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    (void)ir.LoadBufferFormat(ir.Imm32(0U), ir.Imm32(0U), {});

    Shader::Optimization::LowerBufferFormatToRaw(program);

    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::LoadBufferFormatF32), 0U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::LoadBufferU32), 1U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::BitFieldSExtract), 3U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::BitCastF32U32), 3U);

    std::array<bool, 3> found_ranges{};
    for (const Shader::IR::Inst& inst : block->Instructions()) {
        if (inst.GetOpcode() != Shader::IR::Opcode::BitFieldSExtract) {
            continue;
        }
        ASSERT_TRUE(inst.Arg(1).IsImmediate());
        ASSERT_TRUE(inst.Arg(2).IsImmediate());
        const auto offset = inst.Arg(1).U32();
        const auto count = inst.Arg(2).U32();
        if (offset == 0U && count == 11U) {
            found_ranges[0] = true;
        } else if (offset == 11U && count == 11U) {
            found_ranges[1] = true;
        } else if (offset == 22U && count == 10U) {
            found_ranges[2] = true;
        } else {
            ADD_FAILURE() << "Unexpected signed 10_11_11 bit range " << offset << ":" << count;
        }
    }
    EXPECT_EQ(found_ranges, (std::array{true, true, true}));
}

TEST(LowerBufferFormatToRaw, TypedInstructionOverridesDescriptorWithSigned10_11_11) {
    auto info =
        MakeFormattedBufferInfo(AmdGpu::DataFormat::Format32, AmdGpu::NumberFormat::Float);
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    Shader::IR::BufferInstInfo flags{};
    flags.typed.Assign(true);
    flags.inst_data_fmt.Assign(AmdGpu::DataFormat::Format10_11_11);
    flags.inst_num_fmt.Assign(AmdGpu::NumberFormat::Sint);
    (void)ir.LoadBufferFormat(ir.Imm32(0U), ir.Imm32(0U), flags);

    Shader::Optimization::LowerBufferFormatToRaw(program);

    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::LoadBufferF32), 0U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::LoadBufferU32), 1U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::BitFieldSExtract), 3U);
}

TEST(LowerBufferFormatToRaw, PacksSigned10_11_11DescriptorStores) {
    auto info = MakeFormattedBufferInfo(AmdGpu::DataFormat::Format10_11_11,
                                        AmdGpu::NumberFormat::Sint, true);
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    const auto data = ir.CompositeConstruct(ir.Imm32(1.f), ir.Imm32(2.f), ir.Imm32(3.f),
                                            ir.Imm32(0.f));
    ir.StoreBufferFormat(ir.Imm32(0U), ir.Imm32(0U), data, {});

    Shader::Optimization::LowerBufferFormatToRaw(program);

    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::StoreBufferFormatF32), 0U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::StoreBufferU32), 1U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::BitCastU32F32), 3U);
    EXPECT_EQ(CountOpcode(*block, Shader::IR::Opcode::BitFieldInsert), 2U);

    std::array<bool, 2> found_ranges{};
    for (const Shader::IR::Inst& inst : block->Instructions()) {
        if (inst.GetOpcode() != Shader::IR::Opcode::BitFieldInsert) {
            continue;
        }
        ASSERT_TRUE(inst.Arg(2).IsImmediate());
        ASSERT_TRUE(inst.Arg(3).IsImmediate());
        const auto offset = inst.Arg(2).U32();
        const auto count = inst.Arg(3).U32();
        if (offset == 11U && count == 11U) {
            found_ranges[0] = true;
        } else if (offset == 22U && count == 10U) {
            found_ranges[1] = true;
        } else {
            ADD_FAILURE() << "Unexpected signed 10_11_11 insert range " << offset << ":" << count;
        }
    }
    EXPECT_EQ(found_ranges, (std::array{true, true}));
}
