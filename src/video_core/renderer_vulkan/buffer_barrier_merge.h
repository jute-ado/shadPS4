// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <ranges>

#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

template <std::ranges::range Range>
[[nodiscard]] bool TryMergeBufferBarrier(Range& barriers,
                                         const vk::BufferMemoryBarrier2& additional) {
    const auto existing = std::ranges::find_if(barriers, [&](const auto& barrier) {
        return barrier.buffer == additional.buffer && barrier.offset == additional.offset &&
               barrier.size == additional.size &&
               barrier.srcQueueFamilyIndex == additional.srcQueueFamilyIndex &&
               barrier.dstQueueFamilyIndex == additional.dstQueueFamilyIndex;
    });
    if (existing == std::ranges::end(barriers)) {
        return false;
    }
    existing->srcStageMask |= additional.srcStageMask;
    existing->srcAccessMask |= additional.srcAccessMask;
    existing->dstStageMask |= additional.dstStageMask;
    existing->dstAccessMask |= additional.dstAccessMask;
    return true;
}

} // namespace Vulkan
