// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/amdgpu/resource.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/image_layout_policy.h"
#include "video_core/texture_cache/image_usage_policy.h"
#include "video_core/texture_cache/sampler_usage_policy.h"

namespace VideoCore {
namespace {

constexpr vk::ImageUsageFlags RequiredSampledUsage =
    vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
    vk::ImageUsageFlagBits::eSampled;
constexpr vk::ImageUsageFlags PreferredUsage =
    RequiredSampledUsage | vk::ImageUsageFlagBits::eStorage;

TEST(SamplerUsagePolicy, ReservedAnisotropyEncodingsFallBackToOne) {
    EXPECT_EQ(NormalizeSamplerAnisotropy(AmdGpu::AnisoRatio::Sixteen),
              AmdGpu::AnisoRatio::Sixteen);
    EXPECT_EQ(NormalizeSamplerAnisotropy(static_cast<AmdGpu::AnisoRatio>(5)),
              AmdGpu::AnisoRatio::One);
    EXPECT_EQ(NormalizeSamplerAnisotropy(static_cast<AmdGpu::AnisoRatio>(7)),
              AmdGpu::AnisoRatio::One);
}

TEST(SamplerUsagePolicy, ReservedMipFilterEncodingFallsBackToNone) {
    EXPECT_EQ(NormalizeSamplerMipFilter(AmdGpu::MipFilter::Linear),
              AmdGpu::MipFilter::Linear);
    EXPECT_EQ(NormalizeSamplerMipFilter(static_cast<AmdGpu::MipFilter>(3)),
              AmdGpu::MipFilter::None);
}

TEST(ImageResource, NonArrayTexturesIgnoreUndefinedDepthBitsWhenCountingLayers) {
    AmdGpu::Image image{};
    image.type = static_cast<u64>(AmdGpu::ImageType::Color2D);
    image.depth = 5634;
    image.pow2pad = 1;

    EXPECT_EQ(image.NumLayers(), 1U);
}

TEST(ImageResource, ArrayTexturesRetainDepthBasedPowerOfTwoLayerCount) {
    AmdGpu::Image image{};
    image.type = static_cast<u64>(AmdGpu::ImageType::Color2DArray);
    image.depth = 62;
    image.pow2pad = 1;

    EXPECT_EQ(image.NumLayers(), 64U);
}

TEST(ImageResource, InvalidViewTypeFallsBackToTheNullImageDimension) {
    AmdGpu::Image image{};
    image.type = 7;

    EXPECT_FALSE(image.Valid());
    EXPECT_EQ(image.GetShaderViewType(false), AmdGpu::ImageType::Color2D);
    EXPECT_EQ(image.GetShaderViewType(true), AmdGpu::ImageType::Color2D);
}

TEST(ImageResource, ShaderViewTypeRetainsValidArraySemantics) {
    AmdGpu::Image image{};
    image.type = static_cast<u64>(AmdGpu::ImageType::Color2DArray);

    EXPECT_EQ(image.GetShaderViewType(false), AmdGpu::ImageType::Color2D);
    EXPECT_EQ(image.GetShaderViewType(true), AmdGpu::ImageType::Color2DArray);
}

TEST(ImageUsagePolicy, RetainsPreferredUsageWhenTheDeviceSupportsIt) {
    std::vector<vk::ImageUsageFlags> queries;

    const auto result = SelectSupportedImageUsage(
        true, PreferredUsage, [&](vk::ImageUsageFlags usage) {
            queries.push_back(usage);
            return true;
        });

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->usage, PreferredUsage);
    EXPECT_FALSE(result->dropped_optional_storage);
    EXPECT_EQ(queries, std::vector<vk::ImageUsageFlags>{PreferredUsage});
}

TEST(ImageUsagePolicy, DropsOptionalStorageForCompressedImagesWhenRequiredUsageIsSupported) {
    std::vector<vk::ImageUsageFlags> queries;

    const auto result = SelectSupportedImageUsage(
        true, PreferredUsage, [&](vk::ImageUsageFlags usage) {
            queries.push_back(usage);
            return usage == RequiredSampledUsage;
        });

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->usage, RequiredSampledUsage);
    EXPECT_TRUE(result->dropped_optional_storage);
    EXPECT_EQ(queries,
              (std::vector<vk::ImageUsageFlags>{PreferredUsage, RequiredSampledUsage}));
}

TEST(ImageUsagePolicy, DoesNotDropStorageFromUncompressedImages) {
    std::vector<vk::ImageUsageFlags> queries;

    const auto result = SelectSupportedImageUsage(
        false, PreferredUsage, [&](vk::ImageUsageFlags usage) {
            queries.push_back(usage);
            return usage == RequiredSampledUsage;
        });

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(queries, std::vector<vk::ImageUsageFlags>{PreferredUsage});
}

TEST(ImageUsagePolicy, FailsClosedWhenEvenRequiredCompressedUsageIsUnsupported) {
    std::vector<vk::ImageUsageFlags> queries;

    const auto result = SelectSupportedImageUsage(
        true, PreferredUsage, [&](vk::ImageUsageFlags usage) {
            queries.push_back(usage);
            return false;
        });

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(queries,
              (std::vector<vk::ImageUsageFlags>{PreferredUsage, RequiredSampledUsage}));
}

TEST(ImageUsagePolicy, RejectsCorruptArrayDescriptorBeforeAllocatingHostImage) {
    const HostImageResourceLimits limits{
        .max_1d_dimension = 32768,
        .max_2d_dimension = 32768,
        .max_3d_dimension = 16384,
        .max_array_layers = 2048,
        .max_mip_levels = 15,
    };

    EXPECT_FALSE(IsImageResourceWithinHostLimits(
        AmdGpu::ImageType::Color2DArray,
        {.width = 6657, .height = 15367, .depth = 1},
        {.levels = 1, .layers = 32768}, limits));
}

TEST(ImageUsagePolicy, AcceptsValidArrayDescriptorWithinHostLimits) {
    const HostImageResourceLimits limits{
        .max_1d_dimension = 32768,
        .max_2d_dimension = 32768,
        .max_3d_dimension = 16384,
        .max_array_layers = 2048,
        .max_mip_levels = 15,
    };

    EXPECT_TRUE(IsImageResourceWithinHostLimits(
        AmdGpu::ImageType::Color2DArray,
        {.width = 1024, .height = 1024, .depth = 1},
        {.levels = 11, .layers = 64}, limits));
}

TEST(ImageUsagePolicy, RejectsMipZeroWhoseMinimumAllocationExceedsDeviceBudget) {
    EXPECT_FALSE(IsImageEstimatedAllocationWithinLimit(
        {.width = 6248, .height = 15409, .depth = 1}, {.levels = 1, .layers = 2048},
        64, false, 1, 16_GB));
    EXPECT_TRUE(IsImageEstimatedAllocationWithinLimit(
        {.width = 1024, .height = 1024, .depth = 1}, {.levels = 11, .layers = 64},
        128, true, 1, 16_GB));
}

TEST(ImageUsagePolicy, RequiresCompleteGuestBackingForNonSparseImages) {
    EXPECT_TRUE(IsImageGuestBackingUsable(AmdGpu::ArrayMode::Array2DTiledThin1, true));
    EXPECT_FALSE(IsImageGuestBackingUsable(AmdGpu::ArrayMode::Array2DTiledThin1, false));
    EXPECT_TRUE(IsImageGuestBackingUsable(AmdGpu::ArrayMode::ArrayPrt2DTiledThin1, false));
    EXPECT_TRUE(IsImageGuestBackingUsable(AmdGpu::ArrayMode::Array2DTiledThin1, false, true));
}

TEST(ImageUsagePolicy, DiagnosticIncompleteBackingIsLimitedToReadOnlyLodFeedbackImages) {
    EXPECT_TRUE(ShouldAllowDiagnosticIncompleteNonSparseImage(true, false, true));
    EXPECT_FALSE(ShouldAllowDiagnosticIncompleteNonSparseImage(false, false, true));
    EXPECT_FALSE(ShouldAllowDiagnosticIncompleteNonSparseImage(true, true, true));
    EXPECT_FALSE(ShouldAllowDiagnosticIncompleteNonSparseImage(true, false, false));
}

TEST(ImageUsagePolicy, RemembersLodFeedbackAllocationForReadOnlyDescriptorAliases) {
    LodFeedbackImageTracker tracker;
    constexpr VAddr address = 0x407800000;
    constexpr u32 size = 0x2ab400;

    EXPECT_FALSE(tracker.CanUseIncompleteBacking(address, size, false, true));
    tracker.Observe(address, size, true);
    EXPECT_TRUE(tracker.CanUseIncompleteBacking(address, size, false, true));
    EXPECT_TRUE(tracker.CanUseIncompleteBacking(address, size / 2, false, true));
    EXPECT_FALSE(tracker.CanUseIncompleteBacking(address, size + 1, false, true));
    EXPECT_FALSE(tracker.CanUseIncompleteBacking(address, size, true, true));
    EXPECT_FALSE(tracker.CanUseIncompleteBacking(address + 0x10000, size, false, true));
    EXPECT_FALSE(tracker.CanUseIncompleteBacking(address, size, false, false));
}

TEST(ImageUsagePolicy, ForgetsLodFeedbackAllocationWhenItsBackingIsUnmapped) {
    LodFeedbackImageTracker tracker;
    constexpr VAddr address = 0x407800000;
    constexpr u32 size = 0x2ab400;
    tracker.Observe(address, size, true);

    tracker.Invalidate(address + 0x10000, 0x1000);
    EXPECT_FALSE(tracker.CanUseIncompleteBacking(address, size, false, true));
}

TEST(ImageUsagePolicy, RejectsUnsupportedSurfaceFormatBeforeConstructingImageInfo) {
    u32 support_queries = 0;
    EXPECT_FALSE(IsImageDescriptorFormatUsable(
        AmdGpu::DataFormat::Format10_11_11, AmdGpu::NumberFormat::Unorm,
        [&](auto, auto) {
            ++support_queries;
            return false;
        }));
    EXPECT_EQ(support_queries, 1U);

    EXPECT_TRUE(IsImageDescriptorFormatUsable(
        AmdGpu::DataFormat::Format10_11_11, AmdGpu::NumberFormat::Float,
        [](auto, auto) { return true; }));
}

TEST(ImageUsagePolicy, RejectsReservedComponentSwizzlesBeforeConstructingImageView) {
    EXPECT_TRUE(IsImageComponentMappingUsable(0, 1, 4, 7));
    EXPECT_TRUE(IsImageComponentMappingUsable(4, 5, 6, 7));
    EXPECT_FALSE(IsImageComponentMappingUsable(2, 5, 6, 7));
    EXPECT_FALSE(IsImageComponentMappingUsable(4, 5, 6, 3));
    EXPECT_FALSE(IsImageComponentMappingUsable(4, 5, 6, 8));
}

TEST(ImageLayoutPolicy, ClassifiesEveryObservedPrtMacroLayoutWithItsTileThickness) {
    const auto thin = ClassifyImageArrayLayout(
        AmdGpu::ArrayMode::ArrayPrt3DTiledThin1);
    EXPECT_EQ(thin.kind, ImageArrayLayoutKind::MacroTiled);
    EXPECT_EQ(thin.thickness, 1U);

    const auto thick = ClassifyImageArrayLayout(
        AmdGpu::ArrayMode::ArrayPrt3DTiledThick);
    EXPECT_EQ(thick.kind, ImageArrayLayoutKind::MacroTiled);
    EXPECT_EQ(thick.thickness, 4U);

    const auto xthick = ClassifyImageArrayLayout(
        AmdGpu::ArrayMode::Array3DTiledXThick);
    EXPECT_EQ(xthick.kind, ImageArrayLayoutKind::MacroTiled);
    EXPECT_EQ(xthick.thickness, 8U);

    const auto micro = ClassifyImageArrayLayout(
        AmdGpu::ArrayMode::Array1DTiledThick);
    EXPECT_EQ(micro.kind, ImageArrayLayoutKind::MicroTiled);
    EXPECT_EQ(micro.thickness, 4U);
}

TEST(ImageCopyPolicy, UsesBufferForDifferentDepthStencilFormats) {
    EXPECT_TRUE(RequiresBufferCopyForDepthFormatMismatch(
        vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, true, true));
    EXPECT_TRUE(RequiresBufferCopyForDepthFormatMismatch(
        vk::Format::eD16Unorm, vk::Format::eD24UnormS8Uint, true, true));

    EXPECT_FALSE(RequiresBufferCopyForDepthFormatMismatch(
        vk::Format::eD32Sfloat, vk::Format::eD32Sfloat, true, true));
    EXPECT_FALSE(RequiresBufferCopyForDepthFormatMismatch(
        vk::Format::eR32Sfloat, vk::Format::eD32Sfloat, false, true));
    EXPECT_FALSE(RequiresBufferCopyForDepthFormatMismatch(
        vk::Format::eR8G8B8A8Unorm, vk::Format::eB8G8R8A8Unorm, false, false));
}

TEST(AttachmentFeedbackLoopPolicy, AddsRequiredUsageToDepthAndColorAttachments) {
    const auto base = vk::ImageUsageFlagBits::eSampled;
    EXPECT_EQ(WithAttachmentFeedbackLoopUsage(base, true),
              base | vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT);
    EXPECT_EQ(WithAttachmentFeedbackLoopUsage(base, false), base);
}

TEST(AttachmentFeedbackLoopPolicy, SelectsTheWrittenAttachmentAspects) {
    EXPECT_EQ(AttachmentFeedbackLoopAspects(false, false), vk::ImageAspectFlagBits::eColor);
    EXPECT_EQ(AttachmentFeedbackLoopAspects(true, false), vk::ImageAspectFlagBits::eDepth);
    EXPECT_EQ(AttachmentFeedbackLoopAspects(true, true),
              vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil);
}

TEST(ImageResourceMergePolicy, RecalculatesBackingLayoutWhenLayersExpand) {
    struct ResourceExtent {
        std::uint32_t levels;
        std::uint32_t layers;
        auto operator<=>(const ResourceExtent&) const = default;
    };
    struct FakeImageInfo {
        ResourceExtent resources{.levels = 1, .layers = 1};
        std::uint32_t guest_size = 4 * 1024 * 1024;
        std::uint32_t update_count{};

        void UpdateSize() {
            guest_size = resources.layers * 4 * 1024 * 1024;
            ++update_count;
        }
    } info;

    ExpandImageResources(info, ResourceExtent{.levels = 1, .layers = 4});

    EXPECT_EQ(info.resources, (ResourceExtent{.levels = 1, .layers = 4}));
    EXPECT_EQ(info.guest_size, 16 * 1024 * 1024U);
    EXPECT_EQ(info.update_count, 1U);
}

TEST(ImageResourceMergePolicy, DoesNotRecalculateAnUnchangedBackingLayout) {
    struct ResourceExtent {
        std::uint32_t levels;
        std::uint32_t layers;
        auto operator<=>(const ResourceExtent&) const = default;
    };
    struct FakeImageInfo {
        ResourceExtent resources{.levels = 3, .layers = 4};
        std::uint32_t update_count{};

        void UpdateSize() {
            ++update_count;
        }
    } info;

    ExpandImageResources(info, ResourceExtent{.levels = 2, .layers = 4});

    EXPECT_EQ(info.resources, (ResourceExtent{.levels = 3, .layers = 4}));
    EXPECT_EQ(info.update_count, 0U);
}

} // namespace
} // namespace VideoCore
