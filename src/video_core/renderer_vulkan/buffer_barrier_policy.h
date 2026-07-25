// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <ranges>

#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

constexpr vk::AccessFlags2 BufferWriteAccess =
    vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite |
    vk::AccessFlagBits2::eHostWrite | vk::AccessFlagBits2::eMemoryWrite;

constexpr bool HasBufferWriteAccess(vk::AccessFlags2 access) {
    return static_cast<bool>(access & BufferWriteAccess);
}

constexpr bool NeedsBufferBarrier(vk::AccessFlags2 current_access,
                                  vk::PipelineStageFlagBits2 current_stage,
                                  vk::AccessFlags2 destination_access,
                                  vk::PipelineStageFlagBits2 destination_stage) {
    return current_access != destination_access || current_stage != destination_stage ||
           HasBufferWriteAccess(current_access);
}

constexpr vk::AccessFlags2 ShaderBufferAccess(bool is_written) {
    vk::AccessFlags2 access = vk::AccessFlagBits2::eShaderRead;
    if (is_written) {
        access |= vk::AccessFlagBits2::eShaderWrite;
    }
    return access;
}

constexpr vk::AccessFlags2 MergeShaderBufferAccess(vk::AccessFlags2 current, bool is_written) {
    return current | ShaderBufferAccess(is_written);
}

template <std::ranges::range Range>
bool TryMergeBufferBarrierDestination(Range& barriers,
                                      const vk::BufferMemoryBarrier2& additional) {
    const auto existing = std::ranges::find_if(barriers, [&](const auto& barrier) {
        return barrier.buffer == additional.buffer && barrier.offset == additional.offset &&
               barrier.size == additional.size;
    });
    if (existing == std::ranges::end(barriers)) {
        return false;
    }
    existing->dstStageMask |= additional.dstStageMask;
    existing->dstAccessMask |= additional.dstAccessMask;
    return true;
}

} // namespace Vulkan
