// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/liverpool_to_vk.h"

namespace VideoCore {

inline bool ShouldUseAssociatedDepthForView(vk::Format view_format) {
    return Vulkan::LiverpoolToVK::IsFormatStencilCompatible(view_format);
}

} // namespace VideoCore
