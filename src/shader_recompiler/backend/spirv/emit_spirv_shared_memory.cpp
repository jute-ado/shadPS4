// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

Id EmitLoadSharedU16(EmitContext& ctx, Id offset) {
    const Id shift_id{ctx.ConstU32(1U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer = ctx.EmitSharedMemoryAccess(ctx.shared_u16, ctx.shared_memory_u16, index);
    return ctx.OpLoad(ctx.U16, pointer);
}

Id EmitLoadSharedU32(EmitContext& ctx, Id offset) {
    const Id shift_id{ctx.ConstU32(2U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer = ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index);
    return ctx.OpLoad(ctx.U32[1], pointer);
}

Id EmitLoadSharedU64(EmitContext& ctx, Id offset) {
    const Id shift_id{ctx.ConstU32(3U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer = ctx.EmitSharedMemoryAccess(ctx.shared_u64, ctx.shared_memory_u64, index);
    return ctx.OpLoad(ctx.U64, pointer);
}

template <typename Store>
void EmitGuestExecStore(EmitContext& ctx, Id predicate, Store&& store) {
    if (!ctx.info.emulate_wave64_control_flow) {
        store();
        return;
    }
    const Id store_label = ctx.OpLabel();
    const Id merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(predicate, store_label, merge_label);
    ctx.AddLabel(store_label);
    store();
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

void EmitWriteSharedU16(EmitContext& ctx, Id offset, Id value, Id predicate) {
    const Id shift{ctx.ConstU32(1U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift)};
    const Id pointer = ctx.EmitSharedMemoryAccess(ctx.shared_u16, ctx.shared_memory_u16, index);
    EmitGuestExecStore(ctx, predicate, [&] { ctx.OpStore(pointer, value); });
}

void EmitWriteSharedU32(EmitContext& ctx, Id offset, Id value, Id predicate) {
    const Id shift{ctx.ConstU32(2U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift)};
    const Id pointer = ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index);
    EmitGuestExecStore(ctx, predicate, [&] { ctx.OpStore(pointer, value); });
}

void EmitWriteSharedU64(EmitContext& ctx, Id offset, Id value, Id predicate) {
    const Id shift{ctx.ConstU32(3U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift)};
    const Id pointer = ctx.EmitSharedMemoryAccess(ctx.shared_u64, ctx.shared_memory_u64, index);
    EmitGuestExecStore(ctx, predicate, [&] { ctx.OpStore(pointer, value); });
}

} // namespace Shader::Backend::SPIRV
