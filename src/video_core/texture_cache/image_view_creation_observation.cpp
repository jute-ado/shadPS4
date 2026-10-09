// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include "video_core/texture_cache/image_view.h"

namespace VideoCore {
namespace {

[[nodiscard]] constexpr ImageViewFormatClass ClassifyImageViewFormat(vk::Format format) noexcept {
    switch (format) {
    case vk::Format::eUndefined:
        return ImageViewFormatClass::Unknown;
    case vk::Format::eD16Unorm:
    case vk::Format::eX8D24UnormPack32:
    case vk::Format::eD32Sfloat:
        return ImageViewFormatClass::Depth;
    case vk::Format::eS8Uint:
        return ImageViewFormatClass::StencilCompatible;
    case vk::Format::eD16UnormS8Uint:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD32SfloatS8Uint:
        return ImageViewFormatClass::DepthStencil;
    case vk::Format::eR16Unorm:
    case vk::Format::eR32Uint:
    case vk::Format::eR32Sfloat:
        return ImageViewFormatClass::DepthCompatible;
    case vk::Format::eR8Uint:
    case vk::Format::eR8Unorm:
        return ImageViewFormatClass::StencilCompatible;
    default:
        return ImageViewFormatClass::Color;
    }
}

[[nodiscard]] constexpr ImageViewFormatClass
ClassifyResolvedImageViewFormat(vk::Format format) noexcept {
    switch (format) {
    case vk::Format::eUndefined:
        return ImageViewFormatClass::Unknown;
    case vk::Format::eD16Unorm:
    case vk::Format::eX8D24UnormPack32:
    case vk::Format::eD32Sfloat:
        return ImageViewFormatClass::Depth;
    case vk::Format::eS8Uint:
        return ImageViewFormatClass::Stencil;
    case vk::Format::eD16UnormS8Uint:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD32SfloatS8Uint:
        return ImageViewFormatClass::DepthStencil;
    default:
        return ImageViewFormatClass::Color;
    }
}

[[nodiscard]] constexpr ImageViewAspectClass
ClassifyImageViewAspect(vk::ImageAspectFlags aspect) noexcept {
    if (!aspect) {
        return ImageViewAspectClass::None;
    }
    if (aspect == vk::ImageAspectFlagBits::eColor) {
        return ImageViewAspectClass::Color;
    }
    if (aspect == vk::ImageAspectFlagBits::eDepth) {
        return ImageViewAspectClass::Depth;
    }
    if (aspect == vk::ImageAspectFlagBits::eStencil) {
        return ImageViewAspectClass::Stencil;
    }
    if (aspect == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)) {
        return ImageViewAspectClass::DepthStencil;
    }
    return ImageViewAspectClass::Other;
}

[[nodiscard]] constexpr bool IsIdentityComponent(vk::ComponentSwizzle component,
                                                  vk::ComponentSwizzle explicit_component) {
    return component == vk::ComponentSwizzle::eIdentity || component == explicit_component;
}

[[nodiscard]] constexpr ImageViewMappingClass
ClassifyImageViewMapping(const vk::ComponentMapping& mapping) noexcept {
    if (IsIdentityComponent(mapping.r, vk::ComponentSwizzle::eR) &&
        IsIdentityComponent(mapping.g, vk::ComponentSwizzle::eG) &&
        IsIdentityComponent(mapping.b, vk::ComponentSwizzle::eB) &&
        IsIdentityComponent(mapping.a, vk::ComponentSwizzle::eA)) {
        return ImageViewMappingClass::Identity;
    }
    if (mapping.r == vk::ComponentSwizzle::eR && mapping.g == vk::ComponentSwizzle::eR &&
        mapping.b == vk::ComponentSwizzle::eR && mapping.a == vk::ComponentSwizzle::eR) {
        return ImageViewMappingClass::ReplicatedR;
    }
    const auto is_constant = [](vk::ComponentSwizzle component) {
        return component == vk::ComponentSwizzle::eZero || component == vk::ComponentSwizzle::eOne;
    };
    if (is_constant(mapping.r) && is_constant(mapping.g) && is_constant(mapping.b) &&
        is_constant(mapping.a)) {
        return ImageViewMappingClass::ScalarConstants;
    }
    return ImageViewMappingClass::Other;
}

[[nodiscard]] constexpr vk::ImageViewType
ExpectedImageViewType(AmdGpu::ImageType type) noexcept {
    switch (type) {
    case AmdGpu::ImageType::Color1D:
        return vk::ImageViewType::e1D;
    case AmdGpu::ImageType::Color1DArray:
        return vk::ImageViewType::e1DArray;
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
        return vk::ImageViewType::e2D;
    case AmdGpu::ImageType::Color2DArray:
        return vk::ImageViewType::e2DArray;
    case AmdGpu::ImageType::Color3D:
        return vk::ImageViewType::e3D;
    default:
        return static_cast<vk::ImageViewType>(VK_IMAGE_VIEW_TYPE_MAX_ENUM);
    }
}

} // namespace

ImageViewCreationObservation BuildImageViewCreationObservation(
    const ImageViewInfo& requested, vk::Format backing_format,
    vk::ImageAspectFlags backing_aspect, vk::ImageUsageFlags backing_usage,
    vk::Image expected_image, vk::Format expected_resolved_format,
    vk::ImageAspectFlags expected_resolved_aspect, const vk::ImageViewCreateInfo& create_info,
    const vk::ImageViewUsageCreateInfo& usage_info,
    const vk::ImageViewMinLodCreateInfoEXT* min_lod_info, bool created) noexcept {
    vk::ImageUsageFlags expected_usage = backing_usage;
    if (!requested.is_storage) {
        expected_usage &= ~vk::ImageUsageFlagBits::eStorage;
    }

    bool min_lod_exact = true;
    if (min_lod_info != nullptr) {
        min_lod_exact = requested.min_lod != 0 && requested.range.extent.levels != 0;
        if (min_lod_exact) {
            const float last_level = static_cast<float>(requested.range.base.level +
                                                        requested.range.extent.levels - 1);
            const float expected_min_lod =
                std::min(static_cast<float>(requested.min_lod) / 256.f, last_level);
            min_lod_exact = min_lod_info->minLod == expected_min_lod;
        }
    }

    const auto& actual_range = create_info.subresourceRange;
    const bool resolved_aspect_supported =
        expected_resolved_aspect &&
        (expected_resolved_aspect & backing_aspect) == expected_resolved_aspect;
    return {
        .requested_format_class = ClassifyImageViewFormat(requested.format),
        .backing_format_class = ClassifyResolvedImageViewFormat(backing_format),
        .actual_format_class = ClassifyResolvedImageViewFormat(create_info.format),
        .actual_aspect_class = ClassifyImageViewAspect(actual_range.aspectMask),
        .mapping_class = ClassifyImageViewMapping(create_info.components),
        .image_exact = create_info.image == expected_image,
        .view_type_exact = create_info.viewType == ExpectedImageViewType(requested.type),
        .format_exact = create_info.format == expected_resolved_format,
        .aspect_exact = actual_range.aspectMask == expected_resolved_aspect &&
                        resolved_aspect_supported,
        .subresource_exact =
            actual_range.baseMipLevel == requested.range.base.level &&
            actual_range.levelCount == requested.range.extent.levels &&
            actual_range.baseArrayLayer == requested.range.base.layer &&
            actual_range.layerCount == requested.range.extent.layers,
        .mapping_exact = create_info.components == requested.mapping,
        .usage_exact = usage_info.usage == expected_usage,
        .min_lod_exact = min_lod_exact,
        .chain_exact = create_info.pNext == &usage_info && usage_info.pNext == min_lod_info,
        .created = created,
    };
}

} // namespace VideoCore
