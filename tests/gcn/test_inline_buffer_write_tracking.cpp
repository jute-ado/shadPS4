// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <optional>

#include <gtest/gtest.h>

#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

namespace Shader::Optimization {
namespace {

enum class InlineBufferAccess {
    Store,
    FormattedStore,
    Atomic,
};

AmdGpu::Buffer MakeInlineBuffer() {
    AmdGpu::Buffer buffer{};
    buffer.base_address = 1;
    buffer.stride = 4;
    buffer.num_records = 64;
    buffer.data_format = static_cast<u32>(AmdGpu::DataFormat::Format32);
    return buffer;
}

std::optional<BufferResource> TrackInlineBuffer(InlineBufferAccess access) {
    Info info{};
    info.stage = Stage::Compute;

    Pools pools{};
    IR::Program program{info};
    Gcn::Block cfg_block{};
    IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    block->cfg_block = &cfg_block;
    program.blocks.push_back(block);

    IR::IREmitter ir{*block};
    const auto raw = std::bit_cast<std::array<u32, 4>>(MakeInlineBuffer());
    const IR::Value handle = ir.CompositeConstruct(ir.Imm32(raw[0]), ir.Imm32(raw[1]),
                                                   ir.Imm32(raw[2]), ir.Imm32(raw[3]));
    const IR::Value address = ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U));
    switch (access) {
    case InlineBufferAccess::Store:
        ir.StoreBufferU32(1, handle, address, ir.Imm32(0x12345678U), {});
        break;
    case InlineBufferAccess::FormattedStore:
        ir.StoreBufferFormat(handle, address, ir.Imm32(0x12345678U), {});
        break;
    case InlineBufferAccess::Atomic:
        ir.Reference(ir.BufferAtomicIAdd(handle, address, ir.Imm32(1U), {}));
        break;
    }

    ResourceTrackingPass(program, Profile{});
    if (info.buffers.size() != 1) {
        return std::nullopt;
    }
    return info.buffers.front();
}

TEST(InlineBufferWriteTracking, UnformattedStoreMarksInlineDescriptorWritable) {
    const auto resource = TrackInlineBuffer(InlineBufferAccess::Store);

    ASSERT_TRUE(resource.has_value());
    EXPECT_TRUE(resource->inline_cbuf);
    EXPECT_TRUE(resource->is_written);
    EXPECT_FALSE(resource->is_formatted);
}

TEST(InlineBufferWriteTracking, FormattedStoreMarksInlineDescriptorWritableAndFormatted) {
    const auto resource = TrackInlineBuffer(InlineBufferAccess::FormattedStore);

    ASSERT_TRUE(resource.has_value());
    EXPECT_TRUE(resource->inline_cbuf);
    EXPECT_TRUE(resource->is_written);
    EXPECT_TRUE(resource->is_formatted);
}

TEST(InlineBufferWriteTracking, AtomicMarksInlineDescriptorWritable) {
    const auto resource = TrackInlineBuffer(InlineBufferAccess::Atomic);

    ASSERT_TRUE(resource.has_value());
    EXPECT_TRUE(resource->inline_cbuf);
    EXPECT_TRUE(resource->is_written);
    EXPECT_FALSE(resource->is_formatted);
}

} // namespace
} // namespace Shader::Optimization
