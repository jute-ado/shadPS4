// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {

constexpr bool CanCopyImageDirectly(bool maintenance8_supported, bool src_is_depth,
                                    vk::Format src_format, bool dst_is_depth,
                                    vk::Format dst_format) {
    return src_is_depth == dst_is_depth || maintenance8_supported;
}

} // namespace VideoCore
