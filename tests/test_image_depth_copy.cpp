// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "video_core/texture_cache/depth_copy_policy.h"
#include "video_core/texture_cache/image_buffer_transfer_policy.h"
#include "video_core/texture_cache/image_copy_layer_policy.h"

namespace VideoCore {
namespace {

vk::ImageCopy DepthRegion(unsigned width, unsigned height, unsigned layers = 1) {
    vk::ImageCopy copy{};
    copy.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
    copy.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
    copy.srcSubresource.layerCount = layers;
    copy.dstSubresource.layerCount = layers;
    copy.extent.width = width;
    copy.extent.height = height;
    copy.extent.depth = 1;
    return copy;
}

TEST(ImageDepthCopy, DifferentFormatsWithIdenticalDepthPlanesUseBufferInBothDirections) {
    constexpr std::array formats{
        std::pair{vk::Format::eD16Unorm, vk::Format::eD16UnormS8Uint},
        std::pair{vk::Format::eX8D24UnormPack32, vk::Format::eD24UnormS8Uint},
        std::pair{vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint}};
    for (const auto [depth, depth_stencil] : formats) {
        EXPECT_EQ(SelectDepthCopyMode(depth, depth_stencil, 1, 1), DepthCopyMode::Buffer);
        EXPECT_EQ(SelectDepthCopyMode(depth_stencil, depth, 1, 1), DepthCopyMode::Buffer);
    }
}

TEST(ImageDepthCopy, IdenticalFormatsAndExistingColorHandlingRemainDirect) {
    EXPECT_EQ(SelectDepthCopyMode(vk::Format::eD32Sfloat, vk::Format::eD32Sfloat, 1, 1),
              DepthCopyMode::Direct);
    EXPECT_EQ(SelectDepthCopyMode(vk::Format::eD32SfloatS8Uint, vk::Format::eD32SfloatS8Uint, 4, 4),
              DepthCopyMode::Direct);
    EXPECT_EQ(SelectDepthCopyMode(vk::Format::eR32Sfloat, vk::Format::eD32Sfloat, 1, 1),
              DepthCopyMode::Direct);
    EXPECT_EQ(SelectDepthCopyMode(vk::Format::eR8G8B8A8Unorm, vk::Format::eR8G8B8A8Uint, 1, 1),
              DepthCopyMode::Direct);
}

TEST(ImageDepthCopy, EqualByteSizesDoNotAdmitDifferentDepthRepresentations) {
    EXPECT_EQ(SelectDepthCopyMode(vk::Format::eD32Sfloat, vk::Format::eX8D24UnormPack32, 1, 1),
              DepthCopyMode::Unsupported);
    EXPECT_EQ(SelectDepthCopyMode(vk::Format::eD16UnormS8Uint, vk::Format::eD32SfloatS8Uint, 1, 1),
              DepthCopyMode::Unsupported);
    const std::array copies{DepthRegion(1, 1)};
    EXPECT_FALSE(
        BuildDepthCopyBufferPlan(vk::Format::eD32Sfloat, vk::Format::eX8D24UnormPack32, copies));
}

TEST(ImageDepthCopy, BufferTransfersRejectMultisampledBackings) {
    for (const auto [source, destination] :
         {std::pair{1u, 4u}, std::pair{4u, 1u}, std::pair{4u, 4u}}) {
        EXPECT_EQ(SelectDepthCopyMode(vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, source,
                                      destination),
                  DepthCopyMode::Unsupported);
    }
}

TEST(ImageBufferTransfer, BufferTransfersRequireSingleSampleImages) {
    EXPECT_TRUE(SupportsImageBufferTransfer(1));
    for (const u32 samples : {0u, 2u, 4u, 8u, 16u}) {
        EXPECT_FALSE(SupportsImageBufferTransfer(samples));
    }
}

TEST(ImageBufferTransfer, BufferRoundTripRequiresSingleSampleSourceAndDestination) {
    EXPECT_TRUE(SupportsImageBufferRoundTrip(1, 1));
    EXPECT_FALSE(SupportsImageBufferRoundTrip(4, 1));
    EXPECT_FALSE(SupportsImageBufferRoundTrip(1, 4));
    EXPECT_FALSE(SupportsImageBufferRoundTrip(4, 4));
}

TEST(ImageDepthCopy, MipAndArrayOffsetsSurviveDepthPlaneTransfer) {
    auto copy = DepthRegion(3, 5, 2);
    copy.srcSubresource.aspectMask |= vk::ImageAspectFlagBits::eStencil;
    copy.srcSubresource.mipLevel = 0;
    copy.srcSubresource.baseArrayLayer = 1;
    copy.dstSubresource.mipLevel = 3;
    copy.dstSubresource.baseArrayLayer = 4;
    copy.srcOffset.x = 2;
    copy.dstOffset.y = 7;
    const auto plan = BuildDepthCopyBufferPlan(vk::Format::eD32SfloatS8Uint, vk::Format::eD32Sfloat,
                                               std::span{&copy, 1});
    ASSERT_TRUE(plan);
    ASSERT_EQ(plan->sources.size(), 1);
    ASSERT_EQ(plan->destinations.size(), 1);
    EXPECT_EQ(plan->byte_size, 120);
    EXPECT_EQ(plan->sources[0].imageSubresource.aspectMask, vk::ImageAspectFlagBits::eDepth);
    EXPECT_EQ(plan->destinations[0].imageSubresource.aspectMask, vk::ImageAspectFlagBits::eDepth);
    EXPECT_EQ(plan->sources[0].imageSubresource.baseArrayLayer, 1);
    EXPECT_EQ(plan->destinations[0].imageSubresource.baseArrayLayer, 4);
    EXPECT_EQ(plan->destinations[0].imageSubresource.mipLevel, 3);
    EXPECT_EQ(plan->sources[0].imageOffset.x, 2);
    EXPECT_EQ(plan->destinations[0].imageOffset.y, 7);
}

TEST(ImageDepthCopy, MultipleMipsUseNonOverlappingAlignedBufferRanges) {
    std::array copies{DepthRegion(3, 3), DepthRegion(1, 1)};
    copies[1].srcSubresource.mipLevel = copies[1].dstSubresource.mipLevel = 1;
    const auto plan =
        BuildDepthCopyBufferPlan(vk::Format::eD16Unorm, vk::Format::eD16UnormS8Uint, copies);
    ASSERT_TRUE(plan);
    ASSERT_EQ(plan->sources.size(), 2);
    EXPECT_EQ(plan->sources[0].bufferOffset, 0);
    EXPECT_EQ(plan->sources[1].bufferOffset, 20);
    EXPECT_EQ(plan->destinations[1].bufferOffset, 20);
    EXPECT_EQ(plan->byte_size, 22);
    EXPECT_EQ(plan->sources[1].imageSubresource.mipLevel, 1);
}

TEST(ImageDepthCopy, InvalidShapeAndOverflowFailBeforePublishingAPlan) {
    const auto rejected = [](const vk::ImageCopy& copy) {
        return !BuildDepthCopyBufferPlan(vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
                                         std::span{&copy, 1});
    };
    auto copy = DepthRegion(0, 1);
    EXPECT_TRUE(rejected(copy));
    copy = DepthRegion(1, 1, 0);
    EXPECT_TRUE(rejected(copy));
    copy = DepthRegion(1, 1);
    copy.dstSubresource.layerCount = 2;
    EXPECT_TRUE(rejected(copy));
    copy = DepthRegion(1, 1);
    copy.srcOffset.x = -1;
    EXPECT_TRUE(rejected(copy));
    copy = DepthRegion(1, 1);
    copy.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    EXPECT_TRUE(rejected(copy));
    copy = DepthRegion(std::numeric_limits<unsigned>::max(), std::numeric_limits<unsigned>::max());
    EXPECT_TRUE(rejected(copy));
    EXPECT_FALSE(
        BuildDepthCopyBufferPlan(vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, {}));
}

std::string ImageCopySource() {
    std::ifstream source{SHADPS4_IMAGE_COPY_SOURCE_PATH};
    EXPECT_TRUE(source.is_open());
    return {std::istreambuf_iterator<char>{source}, std::istreambuf_iterator<char>{}};
}

std::string TextureCacheSource() {
    std::ifstream source{SHADPS4_TEXTURE_CACHE_SOURCE_PATH};
    EXPECT_TRUE(source.is_open());
    return {std::istreambuf_iterator<char>{source}, std::istreambuf_iterator<char>{}};
}

std::string CopyFunction(const std::string& source, std::string_view name) {
    const auto start = source.find(name);
    EXPECT_NE(start, std::string::npos);
    if (start == std::string::npos) {
        return {};
    }
    const auto end = source.find("\nvoid Image::", start + name.size());
    return source.substr(start, end == std::string::npos ? end : end - start);
}

TEST(ImageDepthCopy, WholeImageRoutesDifferentDepthFormatsBeforeDirectCopy) {
    const auto function = CopyFunction(ImageCopySource(), "void Image::CopyImage(");
    const auto policy = function.find("SelectDepthCopyMode(");
    const auto buffer_copy = function.find("CopyDepthRegionsWithBuffer(");
    const auto direct_copy = function.find("cmdbuf.copyImage(");
    ASSERT_NE(policy, std::string::npos);
    ASSERT_NE(buffer_copy, std::string::npos);
    ASSERT_NE(direct_copy, std::string::npos);
    EXPECT_LT(policy, buffer_copy);
    EXPECT_LT(buffer_copy, direct_copy);
}

TEST(ImageDepthCopy, MipCopyRoutesDifferentDepthFormatsBeforeDirectCopy) {
    const auto function = CopyFunction(ImageCopySource(), "void Image::CopyMip(");
    const auto policy = function.find("SelectDepthCopyMode(");
    const auto buffer_copy = function.find("CopyDepthRegionsWithBuffer(");
    const auto direct_copy = function.find("cmdbuf.copyImage(");
    ASSERT_NE(policy, std::string::npos);
    ASSERT_NE(buffer_copy, std::string::npos);
    ASSERT_NE(direct_copy, std::string::npos);
    EXPECT_LT(policy, buffer_copy);
    EXPECT_LT(buffer_copy, direct_copy);
}

TEST(ImageDepthCopy, ResolveBoundsEqualTypeCopiesToBothViewRanges) {
    const auto function = CopyFunction(ImageCopySource(), "void Image::Resolve(");
    const auto sanitize = function.find("SanitizeCopyLayers(");
    const auto bound = function.find("ClampEqualImageCopyLayersToRanges(");
    ASSERT_NE(sanitize, std::string::npos);
    ASSERT_NE(bound, std::string::npos);
    EXPECT_LT(sanitize, bound);
    EXPECT_NE(function.find("mrt0_range.extent.layers", bound), std::string::npos);
    EXPECT_NE(function.find("mrt1_range.extent.layers", bound), std::string::npos);
}

TEST(ImageDepthCopy, EqualTypeCopyLayerBoundIsSymmetricAndRangeSafe) {
    EXPECT_EQ(ClampEqualImageCopyLayersToRanges({1, 4}, 1, 4), (CopyLayerCounts{1, 1}));
    EXPECT_EQ(ClampEqualImageCopyLayersToRanges({4, 1}, 3, 2), (CopyLayerCounts{1, 1}));
    EXPECT_EQ(ClampEqualImageCopyLayersToRanges({1, 4}, 3, 2), (CopyLayerCounts{1, 1}));
    EXPECT_EQ(ClampEqualImageCopyLayersToRanges({8, 8}, 4, 6), (CopyLayerCounts{4, 4}));
    EXPECT_EQ(ClampEqualImageCopyLayersToRanges({4, 4}, 0, 4), (CopyLayerCounts{0, 0}));
    EXPECT_EQ(ClampEqualImageCopyLayersToRanges({4, 4}, 4, 0), (CopyLayerCounts{0, 0}));
}

TEST(ImageDepthCopy, MipCopyBoundsLayerCountByDestinationBaseLayer) {
    const auto function = CopyFunction(ImageCopySource(), "void Image::CopyMip(");
    const auto sanitize = function.find("SanitizeCopyLayers(");
    const auto destination_range = function.find("destination_range_layers");
    const auto bound = function.find("ClampEqualImageCopyLayersToRanges(");
    const auto image_copy = function.find("const vk::ImageCopy image_copy");
    ASSERT_NE(sanitize, std::string::npos);
    ASSERT_NE(destination_range, std::string::npos);
    ASSERT_NE(bound, std::string::npos);
    ASSERT_NE(image_copy, std::string::npos);
    EXPECT_LT(sanitize, bound);
    EXPECT_LT(sanitize, destination_range);
    EXPECT_LT(destination_range, bound);
    EXPECT_LT(bound, image_copy);
    EXPECT_NE(function.find("slice < info.resources.layers", destination_range),
              std::string::npos);
    EXPECT_NE(function.find("info.resources.layers - slice", destination_range),
              std::string::npos);
    EXPECT_NE(function.find("src_info.resources.layers", bound), std::string::npos);
}

TEST(ImageDepthCopy, MipCopyLayerPolicyPreservesSingleSliceAndRejectsOutOfRangeBase) {
    // The observed Shadow warning is a one-layer source merged into a four-layer backing image.
    // At slice zero, the copy remains one layer; at the final valid slice it remains one; a base
    // layer at the end of the backing resource has no representable copy range.
    EXPECT_EQ(ClampEqualImageCopyLayersToRanges({1, 4}, 1, 4), (CopyLayerCounts{1, 1}));
    EXPECT_EQ(ClampEqualImageCopyLayersToRanges({1, 4}, 1, 1), (CopyLayerCounts{1, 1}));
    EXPECT_EQ(ClampEqualImageCopyLayersToRanges({1, 4}, 1, 0), (CopyLayerCounts{0, 0}));
    EXPECT_EQ(ClampEqualImageCopyLayersToRanges({4, 4}, 4, 1), (CopyLayerCounts{1, 1}));
}

TEST(ImageDepthCopy, TemporaryStorageIsBarrieredAndRetiredAfterCopySubmission) {
    const auto helper = CopyFunction(ImageCopySource(), "static bool CopyDepthRegionsWithBuffer(");
    const auto source_copy = helper.find("cmdbuf.copyImageToBuffer(");
    const auto barrier = helper.find("cmdbuf.pipelineBarrier2(");
    const auto destination_copy = helper.find("cmdbuf.copyBufferToImage(");
    const auto retire = helper.find("scheduler.DeferOperation(");
    ASSERT_NE(source_copy, std::string::npos);
    ASSERT_NE(barrier, std::string::npos);
    ASSERT_NE(destination_copy, std::string::npos);
    ASSERT_NE(retire, std::string::npos);
    EXPECT_LT(source_copy, barrier);
    EXPECT_LT(barrier, destination_copy);
    EXPECT_LT(destination_copy, retire);
    EXPECT_NE(helper.find("[allocator, buffer, allocation]"), std::string::npos);
    EXPECT_EQ(helper.find("scheduler.Wait("), std::string::npos);
    EXPECT_NE(helper.find(".size = plan->byte_size"), std::string::npos);
}

TEST(ImageBufferTransfer, UploadChecksSampleCountBeforeRecordingCopy) {
    const auto function = CopyFunction(ImageCopySource(), "void Image::Upload(");
    const auto policy = function.find("SupportsImageBufferTransfer(info.num_samples)");
    const auto set_backing = function.find("SetBackingSamples(");
    const auto copy = function.find("cmdbuf.copyBufferToImage(");
    ASSERT_NE(policy, std::string::npos);
    ASSERT_NE(set_backing, std::string::npos);
    ASSERT_NE(copy, std::string::npos);
    EXPECT_LT(policy, set_backing);
    EXPECT_LT(policy, copy);
}

TEST(ImageBufferTransfer, DownloadChecksSampleCountBeforeRecordingCopy) {
    const auto function = CopyFunction(ImageCopySource(), "void Image::Download(");
    const auto policy = function.find("SupportsImageBufferTransfer(info.num_samples)");
    const auto set_backing = function.find("SetBackingSamples(");
    const auto copy = function.find("cmdbuf.copyImageToBuffer(");
    ASSERT_NE(policy, std::string::npos);
    ASSERT_NE(set_backing, std::string::npos);
    ASSERT_NE(copy, std::string::npos);
    EXPECT_LT(policy, set_backing);
    EXPECT_LT(policy, copy);
}

TEST(ImageBufferTransfer, ImageRoundTripChecksBothSampleCountsBeforeRecordingCopies) {
    const auto function = CopyFunction(ImageCopySource(), "void Image::CopyImageWithBuffer(");
    const auto policy = function.find("SupportsImageBufferRoundTrip(src_info.num_samples,");
    const auto set_backing = function.find("SetBackingSamples(");
    const auto source_copy = function.find("cmdbuf.copyImageToBuffer(");
    const auto destination_copy = function.find("cmdbuf.copyBufferToImage(");
    ASSERT_NE(policy, std::string::npos);
    ASSERT_NE(set_backing, std::string::npos);
    ASSERT_NE(source_copy, std::string::npos);
    ASSERT_NE(destination_copy, std::string::npos);
    EXPECT_LT(policy, set_backing);
    EXPECT_LT(policy, source_copy);
    EXPECT_LT(policy, destination_copy);
}

TEST(ImageBufferTransfer, DepthRoundTripChecksActualSampleCountBeforePlanningCopy) {
    const auto helper = CopyFunction(ImageCopySource(), "static bool CopyDepthRegionsWithBuffer(");
    const auto guard = helper.find("SupportsImageBufferRoundTrip(");
    const auto plan = helper.find("BuildDepthCopyBufferPlan(");
    const auto source_copy = helper.find("cmdbuf.copyImageToBuffer(");
    const auto destination_copy = helper.find("cmdbuf.copyBufferToImage(");
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(plan, std::string::npos);
    ASSERT_NE(source_copy, std::string::npos);
    ASSERT_NE(destination_copy, std::string::npos);
    EXPECT_LT(guard, plan);
    EXPECT_LT(guard, source_copy);
    EXPECT_LT(guard, destination_copy);
}

TEST(ImageBufferTransfer, TextureCacheDownloadChecksSampleCountBeforeRecordingCopy) {
    const auto source = TextureCacheSource();
    const auto start = source.find("void TextureCache::DownloadImageMemory(");
    ASSERT_NE(start, std::string::npos);
    const auto end = source.find("\nvoid TextureCache::", start + 1);
    const auto function = source.substr(start, end == std::string::npos ? end : end - start);
    const auto policy = function.find("SupportsImageBufferTransfer(image.info.num_samples)");
    const auto set_backing = function.find("image.SetBackingSamples(");
    const auto backing = function.find("image.backing->num_samples");
    const auto copy = function.find("cmdbuf.copyImageToBuffer(");
    ASSERT_NE(policy, std::string::npos);
    ASSERT_NE(set_backing, std::string::npos);
    ASSERT_NE(backing, std::string::npos);
    ASSERT_NE(copy, std::string::npos);
    EXPECT_LT(policy, set_backing);
    EXPECT_LT(policy, backing);
    EXPECT_LT(policy, copy);
}

TEST(ImageDepthCopy, FinalShaderTransitionDoesNotClaimAnUnsupportedTransferAccess) {
    const auto helper = CopyFunction(ImageCopySource(), "static bool CopyDepthRegionsWithBuffer(");
    const auto transition = helper.find("destination.Transit(vk::ImageLayout::eGeneral,");
    ASSERT_NE(transition, std::string::npos);
    const auto end = helper.find(";", transition);
    ASSERT_NE(end, std::string::npos);
    const auto call = helper.substr(transition, end - transition);
    EXPECT_NE(call.find("vk::AccessFlagBits2::eShaderRead"), std::string::npos);
    EXPECT_EQ(call.find("vk::AccessFlagBits2::eTransferRead"), std::string::npos);
}

} // namespace
} // namespace VideoCore
