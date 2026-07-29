// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {

constexpr bool ShouldUseAssociatedDepthForView(vk::Format view_format) {
    switch (view_format) {
    case vk::Format::eS8Uint:
    case vk::Format::eR8Uint:
    case vk::Format::eR8Unorm:
        return true;
    default:
        return false;
    }
}

} // namespace VideoCore
