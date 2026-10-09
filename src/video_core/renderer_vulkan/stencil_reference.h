// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/amdgpu/regs_depth.h"

namespace Vulkan {

struct StencilReferencePlan {
    u8 reference{};
    bool uses_operation_value{};
    bool compare_conflict{};
};

[[nodiscard]] constexpr StencilReferencePlan PlanStencilReference(
    AmdGpu::StencilFunc fail, AmdGpu::StencilFunc depth_pass,
    AmdGpu::StencilFunc depth_fail, AmdGpu::CompareFunc compare,
    const AmdGpu::StencilRefMask& ref) noexcept {
    const bool uses_operation_value = fail == AmdGpu::StencilFunc::ReplaceOp ||
                                      depth_pass == AmdGpu::StencilFunc::ReplaceOp ||
                                      depth_fail == AmdGpu::StencilFunc::ReplaceOp;
    const bool compare_uses_reference = compare != AmdGpu::CompareFunc::Always &&
                                        compare != AmdGpu::CompareFunc::Never;
    return {
        .reference = uses_operation_value ? ref.stencil_op_val : ref.stencil_test_val,
        .uses_operation_value = uses_operation_value,
        .compare_conflict = uses_operation_value && compare_uses_reference &&
                            ref.stencil_test_val != ref.stencil_op_val,
    };
}

} // namespace Vulkan
