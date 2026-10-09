// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright © 2023 Skyline Team and Contributors (https://github.com/skyline-emu/)
// Copyright © 2015-2023 The Khronos Group Inc.
// Copyright © 2015-2023 Valve Corporation
// Copyright © 2015-2023 LunarG, Inc.

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {

/// Returns true if the two formats are compatible according to Vulkan's format compatibility rules
bool IsVulkanFormatCompatible(vk::Format base, vk::Format view);

/// Depth/stencil images can only use vkCmdCopyImage with matching depth/stencil formats.
/// Maintenance 8 adds depth/stencil-to-color compatibility; it does not make two different
/// depth/stencil formats compatible with each other.
constexpr bool RequiresBufferCopyForDepthFormatMismatch(vk::Format source,
                                                        vk::Format destination,
                                                        bool source_is_depth,
                                                        bool destination_is_depth) {
    return source_is_depth && destination_is_depth && source != destination;
}

} // namespace VideoCore
