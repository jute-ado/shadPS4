// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/amdgpu/regs_color.h"

namespace Vulkan {

[[nodiscard]] constexpr bool IsMinMaxBlendFunction(
    AmdGpu::BlendControl::BlendFunc function) noexcept {
    using enum AmdGpu::BlendControl::BlendFunc;
    return function == Min || function == Max;
}

// GCN applies the source and destination color factors before a min/max operation. Vulkan's
// min/max blend ops ignore those factors, so the renderer must carry the operation in a squared
// attachment domain and use one/one factors. This predicate identifies that exact self-scaled
// operation; callers must still perform the corresponding shader and attachment-domain rewrite.
[[nodiscard]] constexpr bool IsSelfScaledMinMaxBlend(const AmdGpu::BlendControl& control,
                                                     bool blend_bypass) noexcept {
    using BlendFactor = AmdGpu::BlendControl::BlendFactor;
    return !blend_bypass && control.enable != 0 && IsMinMaxBlendFunction(control.color_func) &&
           control.color_src_factor == BlendFactor::SrcColor &&
           control.color_dst_factor == BlendFactor::DstColor;
}

} // namespace Vulkan
