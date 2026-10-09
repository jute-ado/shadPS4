// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <span>

#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {

template <typename State>
[[nodiscard]] std::optional<vk::ImageLayout> ResolveSubresourceLayout(
    vk::ImageLayout aggregate_layout, std::span<const State> states,
    const SubresourceExtent& available, const SubresourceRange& range) {
    if (!IsValidSubresourceRange(range, available)) {
        return std::nullopt;
    }
    if (states.empty()) {
        return aggregate_layout;
    }
    if (states.size() != u64{available.levels} * available.layers) {
        return std::nullopt;
    }
    const auto first_index = u64{range.base.level} * available.layers + range.base.layer;
    const auto layout = states[first_index].layout;
    for (u32 level = range.base.level; level < range.base.level + range.extent.levels; ++level) {
        for (u32 layer = range.base.layer; layer < range.base.layer + range.extent.layers;
             ++layer) {
            if (states[u64{level} * available.layers + layer].layout != layout) {
                return std::nullopt;
            }
        }
    }
    return layout;
}

} // namespace VideoCore
