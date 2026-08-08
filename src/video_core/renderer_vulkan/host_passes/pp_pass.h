//  SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
//  SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/renderer_vulkan/host_passes/pp_input_shadow.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Frame;
}

namespace Vulkan::HostPasses {

class PostProcessingPass {
public:
    struct Settings {
        float gamma = 1.0f;
        u32 hdr = 0;
    };

    void Create(vk::Device device, vk::Format surface_format, bool enable_input_shadow);

    [[nodiscard]] bool Render(vk::CommandBuffer cmdbuf, vk::ImageView input,
                              vk::Extent2D input_size, Frame& output, Settings settings,
                              vk::ImageView input_shadow = {});

private:
    vk::UniquePipeline pipeline{};
    vk::UniquePipeline input_shadow_pipeline{};
    vk::UniquePipelineLayout pipeline_layout{};
    vk::UniqueDescriptorSetLayout desc_set_layout{};
    vk::UniqueSampler sampler{};
};

} // namespace Vulkan::HostPasses
