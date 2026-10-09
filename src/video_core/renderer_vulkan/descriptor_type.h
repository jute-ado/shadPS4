// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

constexpr vk::DescriptorType ImageDescriptorType(bool is_written) {
    return is_written ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
}

} // namespace Vulkan
