// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <initializer_list>
#include <optional>

#include <gtest/gtest.h>

#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

namespace Shader::Optimization {
namespace {

enum class BufferAccess {
    Load,
    FormattedLoad,
    Store,
    FormattedStore,
    Atomic,
};

AmdGpu::Buffer MakeBuffer(bool add_tid_enable = false) {
    AmdGpu::Buffer buffer{};
    buffer.base_address = 1;
    buffer.stride = 4;
    buffer.num_records = 64;
    buffer.data_format = static_cast<u32>(AmdGpu::DataFormat::Format32);
    buffer.num_format = static_cast<u32>(AmdGpu::NumberFormat::Uint);
    buffer.add_tid_enable = add_tid_enable;
    return buffer;
}

std::optional<BufferResource> TrackBuffer(std::initializer_list<BufferAccess> accesses,
                                          bool inline_descriptor,
                                          Stage stage = Stage::Compute,
                                          LogicalStage logical_stage = LogicalStage::Compute,
                                          bool add_tid_enable = false) {
    Info info{};
    info.stage = stage;
    info.l_stage = logical_stage;

    Pools pools{};
    IR::Program program{info};
    Gcn::Block cfg_block{};
    IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    block->cfg_block = &cfg_block;
    program.blocks.push_back(block);

    IR::IREmitter ir{*block};
    const auto raw = std::bit_cast<std::array<u32, 4>>(MakeBuffer(add_tid_enable));
    info.flattened_ud_buf.assign(raw.begin(), raw.end());
    const auto word = [&](u32 index) -> IR::U32 {
        return inline_descriptor ? ir.Imm32(raw[index])
                                 : ir.GetUserData(static_cast<IR::ScalarReg>(index));
    };
    const IR::Value handle = ir.CompositeConstruct(word(0), word(1), word(2), word(3));
    const IR::Value address = ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U));
    for (const auto access : accesses) {
        switch (access) {
        case BufferAccess::Load:
            ir.Reference(ir.LoadBufferU32(1, handle, address, {}));
            break;
        case BufferAccess::FormattedLoad:
            ir.Reference(ir.LoadBufferFormat(handle, address, {}));
            break;
        case BufferAccess::Store:
            ir.StoreBufferU32(1, handle, address, ir.Imm32(0x12345678U), {});
            break;
        case BufferAccess::FormattedStore:
            ir.StoreBufferFormat(handle, address,
                                 ir.CompositeConstruct(ir.Imm32(1.0F), ir.Imm32(2.0F),
                                                       ir.Imm32(3.0F), ir.Imm32(4.0F)),
                                 {});
            break;
        case BufferAccess::Atomic:
            ir.Reference(ir.BufferAtomicIAdd(handle, address, ir.Imm32(1U), {}));
            break;
        }
    }

    ResourceTrackingPass(program, Profile{}, false);
    if (info.translation_failed || info.buffers.size() != 1) {
        return std::nullopt;
    }
    return info.buffers.front();
}

void ExpectAccessFlags(std::initializer_list<BufferAccess> accesses, bool written, bool formatted) {
    // The instruction's access semantics must not depend on whether its descriptor is encoded
    // in immediate shader operands or supplied through user-data registers.
    for (const bool inline_descriptor : {false, true}) {
        SCOPED_TRACE(inline_descriptor ? "inline descriptor" : "user-data descriptor");
        const auto resource = TrackBuffer(accesses, inline_descriptor);
        ASSERT_TRUE(resource.has_value());
        EXPECT_EQ(bool(resource->inline_cbuf), inline_descriptor);
        EXPECT_EQ(resource->is_written, written);
        EXPECT_EQ(resource->is_formatted, formatted);
    }
}

TEST(InlineBufferWriteTracking, UnformattedLoadRemainsReadOnly) {
    ExpectAccessFlags({BufferAccess::Load}, false, false);
}

TEST(InlineBufferWriteTracking, FormattedLoadRemainsReadOnlyAndFormatted) {
    ExpectAccessFlags({BufferAccess::FormattedLoad}, false, true);
}

TEST(InlineBufferWriteTracking, UnformattedStoreMarksDescriptorWritable) {
    ExpectAccessFlags({BufferAccess::Store}, true, false);
}

TEST(InlineBufferWriteTracking, FormattedStoreMarksDescriptorWritableAndFormatted) {
    ExpectAccessFlags({BufferAccess::FormattedStore}, true, true);
}

TEST(InlineBufferWriteTracking, AtomicMarksDescriptorWritable) {
    ExpectAccessFlags({BufferAccess::Atomic}, true, false);
}

TEST(InlineBufferWriteTracking, AddTidBufferAddressingRemainsAvailableToCompute) {
    for (const bool inline_descriptor : {false, true}) {
        SCOPED_TRACE(inline_descriptor ? "inline descriptor" : "user-data descriptor");
        EXPECT_TRUE(TrackBuffer({BufferAccess::Load}, inline_descriptor, Stage::Compute,
                                LogicalStage::Compute, true));
    }
}

TEST(InlineBufferWriteTracking, AddTidBufferAddressingFailsSafelyOutsideCompute) {
    for (const auto [stage, logical_stage] :
         {std::pair{Stage::Vertex, LogicalStage::Vertex},
          std::pair{Stage::Fragment, LogicalStage::Fragment}}) {
        for (const bool inline_descriptor : {false, true}) {
            SCOPED_TRACE(inline_descriptor ? "inline descriptor" : "user-data descriptor");
            EXPECT_FALSE(TrackBuffer({BufferAccess::Load}, inline_descriptor, stage,
                                     logical_stage, true));
        }
    }
}

TEST(InlineBufferWriteTracking, LoadThenFormattedStoreCombinesAccessFlagsInOneBinding) {
    ExpectAccessFlags({BufferAccess::Load, BufferAccess::FormattedStore}, true, true);
}

TEST(InlineBufferWriteTracking, FormattedStoreThenLoadRetainsAccessFlagsInOneBinding) {
    ExpectAccessFlags({BufferAccess::FormattedStore, BufferAccess::Load}, true, true);
}

} // namespace
} // namespace Shader::Optimization
