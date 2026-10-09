// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/videoout/buffer.h"

namespace VideoCore {

enum class PresentationInputTransfer : u32 {
    Linear,
    SrgbEncodedUnorm,
};

// Eight-bit sRGB display buffers use a native Vulkan sRGB view, which decodes on sampling.
// Vulkan has no corresponding packed 10-bit sRGB format, so that display format needs an
// explicit decode before the host output pass applies its adjustable sRGB encoding.
constexpr PresentationInputTransfer GetPresentationInputTransfer(
    Libraries::VideoOut::PixelFormat format) {
    return format == Libraries::VideoOut::PixelFormat::A2R10G10B10Srgb
               ? PresentationInputTransfer::SrgbEncodedUnorm
               : PresentationInputTransfer::Linear;
}

} // namespace VideoCore
