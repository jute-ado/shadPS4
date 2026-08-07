// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {

struct BufferAccessState {
    vk::PipelineStageFlagBits2 stage{};
    vk::AccessFlags2 access{};

    auto operator<=>(const BufferAccessState&) const = default;
};

struct BufferAccessTransition {
    BufferAccessState src{};
    BufferAccessState dst{};
    bool required{};
};

struct TiledBufferPublicationPlan {
    bool required{};
    BufferAccessTransition before_write{};
    BufferAccessTransition before_read{};
};

constexpr TiledBufferPublicationPlan GetTiledBufferPublicationPlan(const BufferAccessState previous,
                                                                   const bool is_tiled) {
    if (!is_tiled) {
        return {};
    }
    const BufferAccessState compute_write{
        .stage = vk::PipelineStageFlagBits2::eComputeShader,
        .access = vk::AccessFlagBits2::eShaderWrite,
    };
    const BufferAccessState shader_read{
        .stage = vk::PipelineStageFlagBits2::eAllCommands,
        .access = vk::AccessFlagBits2::eShaderRead,
    };
    return {
        .required = true,
        .before_write = {.src = previous, .dst = compute_write, .required = true},
        .before_read = {.src = compute_write, .dst = shader_read, .required = true},
    };
}

} // namespace VideoCore
