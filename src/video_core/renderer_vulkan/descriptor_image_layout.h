// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

struct DescriptorImageLayoutBinding {
    u32 descriptor_index;
    u32 image_index;
};

struct ResolvedDescriptorImage {
    vk::ImageView image_view;
    vk::ImageLayout image_layout;
};

template <typename ImageResolver>
[[nodiscard]] bool FinalizeDescriptorImageInfos(
    std::span<vk::DescriptorImageInfo> image_infos,
    std::span<const DescriptorImageLayoutBinding> bindings, ImageResolver&& resolve_image) {
    std::vector<ResolvedDescriptorImage> resolved;
    resolved.reserve(bindings.size());
    for (const auto& binding : bindings) {
        auto image = resolve_image(binding.image_index);
        if (!image) {
            return false;
        }
        resolved.push_back(*image);
    }
    for (size_t index = 0; index < bindings.size(); ++index) {
        auto& info = image_infos[bindings[index].descriptor_index];
        info.imageView = resolved[index].image_view;
        info.imageLayout = resolved[index].image_layout;
    }
    return true;
}

template <typename LayoutResolver>
void RefreshDescriptorImageLayouts(
    std::span<vk::DescriptorImageInfo> image_infos,
    std::span<const DescriptorImageLayoutBinding> bindings, LayoutResolver&& resolve_layout) {
    for (const auto& binding : bindings) {
        image_infos[binding.descriptor_index].imageLayout = resolve_layout(binding.image_index);
    }
}

} // namespace Vulkan
