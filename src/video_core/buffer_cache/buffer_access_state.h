// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <optional>

#include "common/assert.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {

inline bool IsReadOnlyBufferAccess(vk::AccessFlags2 access) {
    constexpr auto reads = vk::AccessFlagBits2::eIndirectCommandRead |
                           vk::AccessFlagBits2::eIndexRead |
                           vk::AccessFlagBits2::eVertexAttributeRead |
                           vk::AccessFlagBits2::eUniformRead | vk::AccessFlagBits2::eShaderRead |
                           vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eHostRead |
                           vk::AccessFlagBits2::eMemoryRead;
    // Unknown/extension accesses cannot prove the absence of writes.
    return (static_cast<VkAccessFlags2>(access) & ~static_cast<VkAccessFlags2>(reads)) == 0;
}

// Retains dependencies for potentially writing accesses, including equal masks.
// This tracks transitions, not aggregate usage within a command.
struct BufferAccessState {
    vk::AccessFlags2 access_mask;
    vk::PipelineStageFlagBits2 stage;

    std::optional<vk::BufferMemoryBarrier2> GetBarrier(
        vk::Buffer buffer, u64 size_bytes, vk::AccessFlags2 dst_access_mask,
        vk::PipelineStageFlagBits2 dst_stage, u32 offset = 0) {
        if (dst_access_mask == access_mask && stage == dst_stage &&
            IsReadOnlyBufferAccess(access_mask)) {
            return {};
        }
        DEBUG_ASSERT(offset < size_bytes);
        const auto barrier = vk::BufferMemoryBarrier2{
            .srcStageMask = stage,
            .srcAccessMask = access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access_mask,
            .buffer = buffer,
            .offset = offset,
            .size = size_bytes - offset,
        };
        access_mask = dst_access_mask;
        stage = dst_stage;
        return barrier;
    }
};

inline vk::AccessFlags2 ShaderBufferAccess(bool is_written) {
    return is_written ? vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite
                      : vk::AccessFlagBits2::eShaderRead;
}

} // namespace VideoCore
