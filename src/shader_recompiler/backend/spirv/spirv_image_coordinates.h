// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "shader_recompiler/ir/type.h"
#include "video_core/amdgpu/resource.h"

namespace Shader::Backend::SPIRV {

/// Returns the number of scalar coordinate components required by a sampled image view.
///
/// The array slice is part of the coordinate for array views. Cube views are represented by the
/// recompiler as Color2DArray and therefore require three components as well.
[[nodiscard]] constexpr u32 ImageCoordinateComponentCount(AmdGpu::ImageType view_type) noexcept {
    switch (view_type) {
    case AmdGpu::ImageType::Color1D:
        return 1;
    case AmdGpu::ImageType::Color1DArray:
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
        return 2;
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Color2DMsaaArray:
    case AmdGpu::ImageType::Color3D:
        return 3;
    default:
        return 0;
    }
}

/// Returns the number of scalar components represented by an IR coordinate value.
[[nodiscard]] constexpr u32 CoordinateComponentCount(IR::Type type) noexcept {
    switch (type) {
    case IR::Type::U8:
    case IR::Type::U16:
    case IR::Type::U32:
    case IR::Type::U64:
    case IR::Type::F16:
    case IR::Type::F32:
    case IR::Type::F64:
        return 1;
    case IR::Type::U32x2:
    case IR::Type::F16x2:
    case IR::Type::F32x2:
    case IR::Type::F64x2:
        return 2;
    case IR::Type::U32x3:
    case IR::Type::F16x3:
    case IR::Type::F32x3:
    case IR::Type::F64x3:
        return 3;
    case IR::Type::U32x4:
    case IR::Type::F16x4:
    case IR::Type::F32x4:
    case IR::Type::F64x4:
        return 4;
    default:
        return 0;
    }
}

/// Checks only arity. The SPIR-V validator accepts coordinate values carrying additional
/// components for image operations that consume a prefix (for example a shadow/projector value),
/// so callers must not require exact equality here.
[[nodiscard]] constexpr bool IsImageCoordinateArityValid(AmdGpu::ImageType view_type,
                                                         IR::Type coordinate_type) noexcept {
    const u32 required = ImageCoordinateComponentCount(view_type);
    const u32 actual = CoordinateComponentCount(coordinate_type);
    return required != 0 && actual >= required;
}

/// Checks the complete coordinate type used by the SPIR-V image instruction families emitted by
/// this backend. Sampling/query operations use 32-bit floating-point coordinates; fetch/read,
/// write, and texel-pointer operations use 32-bit unsigned integer coordinates. Keeping this
/// separate from the arity helper lets diagnostics still report the component count while
/// preventing an otherwise well-sized value of the wrong numeric class from reaching SPIR-V.
[[nodiscard]] constexpr bool IsImageCoordinateTypeValid(AmdGpu::ImageType view_type,
                                                        IR::Type coordinate_type,
                                                        bool integer_coordinates) noexcept {
    if (!IsImageCoordinateArityValid(view_type, coordinate_type)) {
        return false;
    }
    if (integer_coordinates) {
        switch (coordinate_type) {
        case IR::Type::U32:
        case IR::Type::U32x2:
        case IR::Type::U32x3:
        case IR::Type::U32x4:
            return true;
        default:
            return false;
        }
    }
    switch (coordinate_type) {
    case IR::Type::F32:
    case IR::Type::F32x2:
    case IR::Type::F32x3:
    case IR::Type::F32x4:
        return true;
    default:
        return false;
    }
}

/// Multisampled images are accessed with image fetch/read plus an explicit sample operand. The
/// sampled-image instructions (implicit/explicit LOD, gather, dref, gradients, and query LOD)
/// are invalid for an MS image in SPIR-V and must be rejected before emission.
[[nodiscard]] constexpr bool IsImageSampleOperationAllowed(AmdGpu::ImageType view_type) noexcept {
    return view_type != AmdGpu::ImageType::Color2DMsaa &&
           view_type != AmdGpu::ImageType::Color2DMsaaArray;
}

/// Returns true when the view has multiple samples and therefore uses the multisampled image
/// operand in SPIR-V. This is deliberately independent of whether the image is sampled or
/// storage-backed; only storage-backed multisampled images require the extra capability.
[[nodiscard]] constexpr bool IsImageMultisampled(AmdGpu::ImageType view_type) noexcept {
    return view_type == AmdGpu::ImageType::Color2DMsaa ||
           view_type == AmdGpu::ImageType::Color2DMsaaArray;
}

/// Storage images use Sampled=2 in OpTypeImage. When that image is multisampled, SPIR-V and
/// Vulkan require both the StorageImageMultisample capability and the corresponding device
/// feature. Keep this policy in one small helper so translation and capability emission cannot
/// drift apart.
[[nodiscard]] constexpr bool RequiresStorageImageMultisampleCapability(
    AmdGpu::ImageType view_type, bool is_storage) noexcept {
    return is_storage && IsImageMultisampled(view_type);
}

[[nodiscard]] constexpr bool IsStorageImageMultisampleAllowed(AmdGpu::ImageType view_type,
                                                              bool is_storage,
                                                              bool feature_supported) noexcept {
    return !RequiresStorageImageMultisampleCapability(view_type, is_storage) ||
           feature_supported;
}

} // namespace Shader::Backend::SPIRV
