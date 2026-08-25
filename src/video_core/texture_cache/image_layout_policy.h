// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>

#include "video_core/amdgpu/tiling.h"

namespace VideoCore {

enum class ImageArrayLayoutKind {
    Invalid,
    Linear,
    MicroTiled,
    MacroTiled,
};

struct ImageArrayLayout {
    ImageArrayLayoutKind kind{ImageArrayLayoutKind::Invalid};
    u32 thickness{1};
};

struct ImageMipGeometry {
    u32 pitch;
    u32 height;
};

[[nodiscard]] constexpr ImageMipGeometry ToBlockCompressedMipGeometry(const u32 pitch,
                                                                       const u32 height) {
    return {.pitch = (pitch + 3) / 4, .height = (height + 3) / 4};
}

[[nodiscard]] constexpr ImageMipGeometry FromBlockCompressedMipGeometry(const u32 pitch,
                                                                         const u32 height) {
    return {.pitch = std::max(pitch * 4, 32u), .height = std::max(height * 4, 32u)};
}

[[nodiscard]] constexpr ImageArrayLayout ClassifyImageArrayLayout(
    const AmdGpu::ArrayMode mode) {
    using enum AmdGpu::ArrayMode;
    switch (mode) {
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
    case ArrayPrt3DTiledThick:
    case Array3DTiledThick:
        return {.kind = ImageArrayLayoutKind::MacroTiled, .thickness = 4};
    case Array2DTiledXThick:
    case Array3DTiledXThick:
        return {.kind = ImageArrayLayoutKind::MacroTiled, .thickness = 8};
    case ArrayLinearGeneral:
        return {};
    }
    return {};
}

} // namespace VideoCore
