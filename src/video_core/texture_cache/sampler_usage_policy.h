// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/amdgpu/resource.h"

namespace VideoCore {

[[nodiscard]] constexpr AmdGpu::AnisoRatio NormalizeSamplerAnisotropy(
    const AmdGpu::AnisoRatio ratio) noexcept {
    return ratio <= AmdGpu::AnisoRatio::Sixteen ? ratio : AmdGpu::AnisoRatio::One;
}

[[nodiscard]] constexpr AmdGpu::MipFilter NormalizeSamplerMipFilter(
    const AmdGpu::MipFilter filter) noexcept {
    return filter <= AmdGpu::MipFilter::Linear ? filter : AmdGpu::MipFilter::None;
}

} // namespace VideoCore
