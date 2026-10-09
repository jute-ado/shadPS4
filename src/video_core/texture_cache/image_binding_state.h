// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {

struct ImageBindingState {
    u32 is_bound : 1;
    u32 is_target : 1;
    u32 needs_rebind : 1;
    u32 force_general : 1;

    constexpr void RecordShaderAccess(bool is_storage) {
        // Storage access is a property of the whole binding batch, not the order in which
        // descriptors were discovered. A later sampled view must not erase an earlier writer.
        force_general |= is_storage;
        is_bound = 1U;
    }

    [[nodiscard]] constexpr vk::AccessFlags2 GeneralLayoutAccessMask(bool is_depth) const {
        vk::AccessFlags2 access = vk::AccessFlagBits2::eShaderRead;
        if (force_general) {
            access |= vk::AccessFlagBits2::eShaderWrite;
        }
        if (is_target) {
            access |= is_depth ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                               : vk::AccessFlagBits2::eColorAttachmentWrite |
                                     vk::AccessFlagBits2::eColorAttachmentRead;
        }
        return access;
    }
};

} // namespace VideoCore
