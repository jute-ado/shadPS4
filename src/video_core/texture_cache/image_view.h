// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/amdgpu/regs_depth.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/types.h"

namespace AmdGpu {
struct ColorBuffer;
}

namespace Shader {
struct ImageResource;
}

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

namespace VideoCore {

struct ImageViewInfo {
    ImageViewInfo() = default;
    ImageViewInfo(const AmdGpu::Image& image, const Shader::ImageResource& desc) noexcept;
    ImageViewInfo(const AmdGpu::ColorBuffer& col_buffer) noexcept;
    ImageViewInfo(const AmdGpu::DepthBuffer& depth_buffer, AmdGpu::DepthView view,
                  AmdGpu::DepthControl ctl);

    AmdGpu::ImageType type = AmdGpu::ImageType::Color2D;
    vk::Format format = vk::Format::eR8G8B8A8Unorm;
    SubresourceRange range;
    vk::ComponentMapping mapping{};
    u32 min_lod = 0;
    bool is_storage = false;

    auto operator<=>(const ImageViewInfo&) const = default;
};

enum class ImageViewFormatClass : u8 {
    Unknown,
    Color,
    DepthCompatible,
    StencilCompatible,
    Depth,
    Stencil,
    DepthStencil,
    Unsupported,
};

enum class ImageViewAspectClass : u8 {
    None,
    Color,
    Depth,
    Stencil,
    DepthStencil,
    Other,
};

enum class ImageViewMappingClass : u8 {
    Identity,
    ReplicatedR,
    ScalarConstants,
    Other,
};

struct ImageViewCreationObservation {
    ImageViewFormatClass requested_format_class{ImageViewFormatClass::Unknown};
    ImageViewFormatClass backing_format_class{ImageViewFormatClass::Unknown};
    ImageViewFormatClass actual_format_class{ImageViewFormatClass::Unknown};
    ImageViewAspectClass actual_aspect_class{ImageViewAspectClass::None};
    ImageViewMappingClass mapping_class{ImageViewMappingClass::Other};
    bool image_exact{};
    bool view_type_exact{};
    bool format_exact{};
    bool aspect_exact{};
    bool subresource_exact{};
    bool mapping_exact{};
    bool usage_exact{};
    bool min_lod_exact{};
    bool chain_exact{};
    bool created{};

    [[nodiscard]] constexpr bool Exact() const noexcept {
        return image_exact && view_type_exact && format_exact && aspect_exact &&
               subresource_exact && mapping_exact && usage_exact && min_lod_exact && chain_exact &&
               created;
    }

    auto operator<=>(const ImageViewCreationObservation&) const = default;
};

[[nodiscard]] ImageViewCreationObservation BuildImageViewCreationObservation(
    const ImageViewInfo& requested, vk::Format backing_format,
    vk::ImageAspectFlags backing_aspect, vk::ImageUsageFlags backing_usage,
    vk::Image expected_image, vk::Format expected_resolved_format,
    vk::ImageAspectFlags expected_resolved_aspect, const vk::ImageViewCreateInfo& create_info,
    const vk::ImageViewUsageCreateInfo& usage_info,
    const vk::ImageViewMinLodCreateInfoEXT* min_lod_info, bool created) noexcept;

struct Image;

struct ImageView {
    ImageView(const Vulkan::Instance& instance, const ImageViewInfo& info, const Image& image);
    ~ImageView();

    ImageView(const ImageView&) = delete;
    ImageView& operator=(const ImageView&) = delete;

    ImageView(ImageView&&) = default;
    ImageView& operator=(ImageView&&) = default;

    [[nodiscard]] const ImageViewCreationObservation& CreationObservation() const noexcept {
        return creation_observation;
    }

    ImageViewInfo info;
    vk::UniqueImageView image_view;

private:
    ImageViewCreationObservation creation_observation{};
};

} // namespace VideoCore
