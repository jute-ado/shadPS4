// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/amdgpu/tiling.h"

namespace VideoCore {

enum class ImageArrayLayoutKind {
    Linear,
    MicroTiled,
    MacroTiled,
};

struct ImageArrayLayout {
    ImageArrayLayoutKind kind{};
    u32 thickness{1};
};

[[nodiscard]] constexpr ImageArrayLayout ClassifyImageArrayLayout(
    const AmdGpu::ArrayMode mode) {
    using enum AmdGpu::ArrayMode;
    switch (mode) {
    case ArrayLinearGeneral:
    case ArrayLinearAligned:
        return {.kind = ImageArrayLayoutKind::Linear, .thickness = 1};
    case Array1DTiledThin1:
        return {.kind = ImageArrayLayoutKind::MicroTiled, .thickness = 1};
    case Array1DTiledThick:
        return {.kind = ImageArrayLayoutKind::MicroTiled, .thickness = 4};
    case Array2DTiledThin1:
    case ArrayPrtTiledThin1:
    case ArrayPrt2DTiledThin1:
    case ArrayPrt3DTiledThin1:
    case Array3DTiledThin1:
        return {.kind = ImageArrayLayoutKind::MacroTiled, .thickness = 1};
    case Array2DTiledThick:
    case ArrayPrtTiledThick:
    case ArrayPrt2DTiledThick:
    case Array3DTiledThick:
    case ArrayPrt3DTiledThick:
        return {.kind = ImageArrayLayoutKind::MacroTiled, .thickness = 4};
    case Array2DTiledXThick:
    case Array3DTiledXThick:
        return {.kind = ImageArrayLayoutKind::MacroTiled, .thickness = 8};
    }
    return {};
}

} // namespace VideoCore
