// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

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

} // namespace Vulkan
