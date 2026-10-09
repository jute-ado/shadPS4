// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/backend/spirv/spirv_image_coordinates.h"

namespace Shader::Backend::SPIRV {
namespace {

TEST(ImageCoordinateContract, ReportsRequiredComponentsForEverySupportedView) {
    EXPECT_EQ(ImageCoordinateComponentCount(AmdGpu::ImageType::Color1D), 1U);
    EXPECT_EQ(ImageCoordinateComponentCount(AmdGpu::ImageType::Color1DArray), 2U);
    EXPECT_EQ(ImageCoordinateComponentCount(AmdGpu::ImageType::Color2D), 2U);
    EXPECT_EQ(ImageCoordinateComponentCount(AmdGpu::ImageType::Color2DArray), 3U);
    EXPECT_EQ(ImageCoordinateComponentCount(AmdGpu::ImageType::Color2DMsaa), 2U);
    EXPECT_EQ(ImageCoordinateComponentCount(AmdGpu::ImageType::Color2DMsaaArray), 3U);
    EXPECT_EQ(ImageCoordinateComponentCount(AmdGpu::ImageType::Color3D), 3U);
    EXPECT_EQ(ImageCoordinateComponentCount(AmdGpu::ImageType::Invalid), 0U);
}

TEST(ImageCoordinateContract, RejectsScalarForMultiComponentView) {
    EXPECT_TRUE(IsImageCoordinateArityValid(AmdGpu::ImageType::Color1D, IR::Type::F32));
    EXPECT_FALSE(IsImageCoordinateArityValid(AmdGpu::ImageType::Color1DArray, IR::Type::F32));
    EXPECT_FALSE(IsImageCoordinateArityValid(AmdGpu::ImageType::Color2D, IR::Type::F32));
    EXPECT_FALSE(IsImageCoordinateArityValid(AmdGpu::ImageType::Color3D, IR::Type::F32x2));
}

TEST(ImageCoordinateContract, AcceptsTheRequiredVectorWidth) {
    EXPECT_TRUE(IsImageCoordinateArityValid(AmdGpu::ImageType::Color1DArray, IR::Type::F32x2));
    EXPECT_TRUE(IsImageCoordinateArityValid(AmdGpu::ImageType::Color2D, IR::Type::U32x2));
    EXPECT_TRUE(IsImageCoordinateArityValid(AmdGpu::ImageType::Color2DArray, IR::Type::F32x3));
    EXPECT_TRUE(IsImageCoordinateArityValid(AmdGpu::ImageType::Color3D, IR::Type::U32x4));
}

TEST(ImageCoordinateContract, RejectsUnknownOrEmptyCoordinateTypes) {
    EXPECT_FALSE(IsImageCoordinateArityValid(AmdGpu::ImageType::Color1D, IR::Type::Void));
    EXPECT_FALSE(IsImageCoordinateArityValid(AmdGpu::ImageType::Invalid, IR::Type::F32));
    EXPECT_EQ(CoordinateComponentCount(IR::Type::Opaque), 0U);
}

TEST(ImageCoordinateContract, MatchesSpirvOperandClassForSampledAndFetchOperations) {
    // OpImageSample* takes floating-point coordinates, while OpImageFetch/Read/Write and
    // OpImageTexelPointer take integer coordinates. Arity alone is insufficient: the old helper
    // accepted both forms and could let malformed IR reach SPIR-V validation.
    EXPECT_TRUE(IsImageCoordinateTypeValid(AmdGpu::ImageType::Color2D, IR::Type::F32x2, false));
    EXPECT_FALSE(IsImageCoordinateTypeValid(AmdGpu::ImageType::Color2D, IR::Type::U32x2, false));
    EXPECT_TRUE(IsImageCoordinateTypeValid(AmdGpu::ImageType::Color2D, IR::Type::U32x2, true));
    EXPECT_FALSE(IsImageCoordinateTypeValid(AmdGpu::ImageType::Color2D, IR::Type::F32x2, true));
    EXPECT_FALSE(IsImageCoordinateTypeValid(AmdGpu::ImageType::Color2D, IR::Type::F16x2, false));
    EXPECT_FALSE(IsImageCoordinateTypeValid(AmdGpu::ImageType::Color2D, IR::Type::F16x2, true));
}

TEST(ImageCoordinateContract, MultisampledViewsRequireFetchOrRead) {
    EXPECT_TRUE(IsImageSampleOperationAllowed(AmdGpu::ImageType::Color1D));
    EXPECT_TRUE(IsImageSampleOperationAllowed(AmdGpu::ImageType::Color2D));
    EXPECT_FALSE(IsImageSampleOperationAllowed(AmdGpu::ImageType::Color2DMsaa));
    EXPECT_FALSE(IsImageSampleOperationAllowed(AmdGpu::ImageType::Color2DMsaaArray));
}

TEST(ImageCoordinateContract, StorageMultisampleCapabilityIsOnlyRequiredForStorageViews) {
    EXPECT_FALSE(RequiresStorageImageMultisampleCapability(AmdGpu::ImageType::Color2D, false));
    EXPECT_FALSE(RequiresStorageImageMultisampleCapability(AmdGpu::ImageType::Color2D, true));
    EXPECT_FALSE(RequiresStorageImageMultisampleCapability(AmdGpu::ImageType::Color2DMsaa, false));
    EXPECT_TRUE(RequiresStorageImageMultisampleCapability(AmdGpu::ImageType::Color2DMsaa, true));
    EXPECT_FALSE(
        RequiresStorageImageMultisampleCapability(AmdGpu::ImageType::Color2DMsaaArray, false));
    EXPECT_TRUE(
        RequiresStorageImageMultisampleCapability(AmdGpu::ImageType::Color2DMsaaArray, true));
}

TEST(ImageCoordinateContract, UnsupportedStorageMultisampleFailsClosedButSampledFetchRemainsValid) {
    for (const auto view_type :
         {AmdGpu::ImageType::Color2DMsaa, AmdGpu::ImageType::Color2DMsaaArray}) {
        EXPECT_FALSE(IsStorageImageMultisampleAllowed(view_type, true, false));
        EXPECT_TRUE(IsStorageImageMultisampleAllowed(view_type, true, true));
        // A sampled MSAA image is accessed through fetch/read and does not require the storage
        // image capability. Keep this distinction explicit because Shadow's malformed resource
        // may be encountered in either path.
        EXPECT_TRUE(IsStorageImageMultisampleAllowed(view_type, false, false));
    }
    EXPECT_TRUE(IsStorageImageMultisampleAllowed(AmdGpu::ImageType::Color2D, true, false));
}

} // namespace
} // namespace Shader::Backend::SPIRV
