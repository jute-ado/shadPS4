// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <optional>

#include "shader_recompiler/ir/operand_helper.h"
#include "shader_recompiler/ir/passes/ir_passes.h"

namespace Shader::Optimization {
namespace {

struct AccessLayout {
    u32 address_shift;
    u32 byte_size;
};

std::optional<AccessLayout> GetAccessLayout(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::ReadConstBuffer:
        return AccessLayout{.address_shift = 2, .byte_size = 4};
    case IR::Opcode::LoadBufferU8:
    case IR::Opcode::StoreBufferU8:
        return AccessLayout{.address_shift = 0, .byte_size = 1};
    case IR::Opcode::LoadBufferU16:
    case IR::Opcode::StoreBufferU16:
        return AccessLayout{.address_shift = 1, .byte_size = 2};
    case IR::Opcode::LoadBufferU32:
    case IR::Opcode::LoadBufferF32:
    case IR::Opcode::StoreBufferU32:
    case IR::Opcode::StoreBufferF32:
    case IR::Opcode::BufferAtomicIAdd32:
    case IR::Opcode::BufferAtomicISub32:
    case IR::Opcode::BufferAtomicSMin32:
    case IR::Opcode::BufferAtomicUMin32:
    case IR::Opcode::BufferAtomicFMin32:
    case IR::Opcode::BufferAtomicSMax32:
    case IR::Opcode::BufferAtomicUMax32:
    case IR::Opcode::BufferAtomicFMax32:
    case IR::Opcode::BufferAtomicInc32:
    case IR::Opcode::BufferAtomicDec32:
    case IR::Opcode::BufferAtomicAnd32:
    case IR::Opcode::BufferAtomicOr32:
    case IR::Opcode::BufferAtomicXor32:
    case IR::Opcode::BufferAtomicSwap32:
    case IR::Opcode::BufferAtomicCmpSwap32:
    case IR::Opcode::BufferAtomicFCmpSwap32:
        return AccessLayout{.address_shift = 2, .byte_size = 4};
    case IR::Opcode::LoadBufferU32x2:
    case IR::Opcode::LoadBufferF32x2:
    case IR::Opcode::StoreBufferU32x2:
    case IR::Opcode::StoreBufferF32x2:
        return AccessLayout{.address_shift = 2, .byte_size = 8};
    case IR::Opcode::LoadBufferU32x3:
    case IR::Opcode::LoadBufferF32x3:
    case IR::Opcode::StoreBufferU32x3:
    case IR::Opcode::StoreBufferF32x3:
        return AccessLayout{.address_shift = 2, .byte_size = 12};
    case IR::Opcode::LoadBufferU32x4:
    case IR::Opcode::LoadBufferF32x4:
    case IR::Opcode::StoreBufferU32x4:
    case IR::Opcode::StoreBufferF32x4:
        return AccessLayout{.address_shift = 2, .byte_size = 16};
    case IR::Opcode::LoadBufferU64:
    case IR::Opcode::StoreBufferU64:
    case IR::Opcode::BufferAtomicIAdd64:
    case IR::Opcode::BufferAtomicSMin64:
    case IR::Opcode::BufferAtomicUMin64:
    case IR::Opcode::BufferAtomicSMax64:
    case IR::Opcode::BufferAtomicUMax64:
        return AccessLayout{.address_shift = 3, .byte_size = 8};
    default:
        return std::nullopt;
    }
}

} // Anonymous namespace

void BufferAccessRangePass(IR::Program& program) {
    auto& info = program.info;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            const auto layout = GetAccessLayout(inst.GetOpcode());
            if (!layout) {
                continue;
            }

            const IR::Value handle = inst.Arg(IR::LoadBufferArgs::Handle);
            if (!handle.IsImmediate() || handle.U32() >= info.buffers.size()) {
                continue;
            }

            auto& resource = info.buffers[handle.U32()];
            if (resource.IsSpecial()) {
                continue;
            }

            const AmdGpu::Buffer sharp = resource.GetSharp(info);
            resource.access_range.RecordAddressLayout(BufferAddressLayout::From(sharp));

            const IR::Value address = inst.Arg(IR::LoadBufferArgs::Address);
            if (!address.IsImmediate()) {
                resource.access_range.MarkDynamic();
                continue;
            }

            const u64 byte_offset = u64{address.U32()} << layout->address_shift;
            resource.access_range.Add(byte_offset, layout->byte_size);
        }
    }
}

} // namespace Shader::Optimization

