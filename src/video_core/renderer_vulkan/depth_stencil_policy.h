// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/amdgpu/regs_depth.h"

namespace VideoCore::DepthStencilPolicy {

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

} // namespace VideoCore::DepthStencilPolicy
