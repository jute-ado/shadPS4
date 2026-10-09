// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

struct DescriptorImageLayoutBinding {
    u32 descriptor_index;
    u32 image_index;
};

template <typename LayoutResolver>
void RefreshDescriptorImageLayouts(
    std::span<vk::DescriptorImageInfo> image_infos,
    std::span<const DescriptorImageLayoutBinding> bindings, LayoutResolver&& resolve_layout) {
    for (const auto& binding : bindings) {
        image_infos[binding.descriptor_index].imageLayout = resolve_layout(binding.image_index);
    }
}

} // namespace Vulkan
