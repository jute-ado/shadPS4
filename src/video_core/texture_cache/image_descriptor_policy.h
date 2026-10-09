// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>

#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/texture_cache/image_binding_policy.h"

namespace VideoCore {

[[nodiscard]] inline std::optional<ImageBindingKind> ClassifyImageBinding(
    const AmdGpu::Image& image) {
    if (image.IsCube()) {
        return ImageBindingKind::Cube;
    }
    switch (image.GetType()) {
    case AmdGpu::ImageType::Color1D:
    case AmdGpu::ImageType::Color1DArray:
        return ImageBindingKind::OneDimensional;
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Color2DMsaa:
    case AmdGpu::ImageType::Color2DMsaaArray:
        return ImageBindingKind::TwoDimensional;
    case AmdGpu::ImageType::Color3D:
        return ImageBindingKind::ThreeDimensional;
    case AmdGpu::ImageType::Cube:
        return ImageBindingKind::Cube;
    case AmdGpu::ImageType::Invalid:
        return std::nullopt;
    }
    return std::nullopt;
}

[[nodiscard]] inline ImageBindingExtent DescribeImageBinding(const AmdGpu::Image& image) {
    return {
        .width = static_cast<u32>(image.width) + 1,
        .height = static_cast<u32>(image.height) + 1,
        .depth = image.GetType() == AmdGpu::ImageType::Color3D
                     ? static_cast<u32>(image.depth) + 1
                     : 1,
        .layers = image.NumLayers(),
        .levels = image.NumLevels(),
    };
}

[[nodiscard]] inline vk::Format DescribeImageBindingFormat(const AmdGpu::Image& image,
                                                           const bool is_depth,
                                                           const bool is_storage) {
    auto number_format = image.GetNumberFmt();
    if (is_storage && number_format == AmdGpu::NumberFormat::Srgb) {
        number_format = AmdGpu::NumberFormat::Unorm;
    }
    auto format = Vulkan::LiverpoolToVK::SurfaceFormat(image.GetDataFmt(), number_format);
    return is_depth ? Vulkan::LiverpoolToVK::PromoteFormatToDepth(format) : format;
}

[[nodiscard]] inline bool IsImageBindingFormatKnown(const AmdGpu::Image& image,
                                                    const bool is_storage) {
    auto number_format = image.GetNumberFmt();
    if (is_storage && number_format == AmdGpu::NumberFormat::Srgb) {
        number_format = AmdGpu::NumberFormat::Unorm;
    }
    return Vulkan::LiverpoolToVK::IsSurfaceFormatKnown(image.GetDataFmt(), number_format);
}

} // namespace VideoCore
