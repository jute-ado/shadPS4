// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/amdgpu/regs_depth.h"

namespace VideoCore::DepthStencilPolicy {

enum class TargetLayout {
    DepthAttachment,
    DepthStencilAttachment,
    DepthReadStencilAttachment,
    DepthStencilReadOnly,
    DepthReadOnly,
};

[[nodiscard]] constexpr bool ShouldClearDepth(bool register_clear_enabled, bool depth_enabled,
                                              bool depth_write_enabled,
                                              bool metadata_cleared) noexcept {
    return metadata_cleared ||
           (register_clear_enabled && depth_enabled && depth_write_enabled);
}

[[nodiscard]] constexpr bool UsesStencilOpValue(AmdGpu::StencilFunc fail,
                                                AmdGpu::StencilFunc depth_pass,
                                                AmdGpu::StencilFunc depth_fail) noexcept {
    return fail == AmdGpu::StencilFunc::ReplaceOp ||
           depth_pass == AmdGpu::StencilFunc::ReplaceOp ||
           depth_fail == AmdGpu::StencilFunc::ReplaceOp;
}

[[nodiscard]] constexpr u8 SelectStencilReference(bool uses_operation_value, u8 operation_value,
                                                   u8 test_value) noexcept {
    return uses_operation_value ? operation_value : test_value;
}

[[nodiscard]] constexpr bool NeedsStencilWriteAccess(
    bool stencil_enabled, bool stencil_clear_enabled, bool backface_enabled,
    const AmdGpu::StencilControl& operations, const AmdGpu::StencilRefMask& front,
    const AmdGpu::StencilRefMask& back) noexcept {
    if (stencil_clear_enabled) {
        return true;
    }
    if (!stencil_enabled) {
        return false;
    }

    const auto face_writes = [](u8 write_mask, AmdGpu::StencilFunc fail,
                                AmdGpu::StencilFunc depth_pass,
                                AmdGpu::StencilFunc depth_fail) {
        return write_mask != 0 &&
               (fail != AmdGpu::StencilFunc::Keep || depth_pass != AmdGpu::StencilFunc::Keep ||
                depth_fail != AmdGpu::StencilFunc::Keep);
    };
    if (face_writes(front.stencil_write_mask, operations.stencil_fail_front,
                    operations.stencil_zpass_front, operations.stencil_zfail_front)) {
        return true;
    }
    return backface_enabled &&
           face_writes(back.stencil_write_mask, operations.stencil_fail_back,
                       operations.stencil_zpass_back, operations.stencil_zfail_back);
}

[[nodiscard]] constexpr TargetLayout SelectTargetLayout(bool depth_write, bool has_stencil,
                                                        bool stencil_write) noexcept {
    if (depth_write) {
        return has_stencil ? TargetLayout::DepthStencilAttachment : TargetLayout::DepthAttachment;
    }
    if (stencil_write) {
        return TargetLayout::DepthReadStencilAttachment;
    }
    return has_stencil ? TargetLayout::DepthStencilReadOnly : TargetLayout::DepthReadOnly;
}

} // namespace VideoCore::DepthStencilPolicy
