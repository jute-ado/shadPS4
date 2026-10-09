// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <tuple>

#include <gtest/gtest.h>

#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/texture_rejection_diagnostics.h"
#include "video_core/texture_cache/image_buffer_transfer_policy.h"
#include "video_core/texture_cache/image_layout_policy.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/tile.h"

namespace {

using Vulkan::TextureRejectionDiagnostics;
using Vulkan::TextureRejectionObservation;

TextureRejectionObservation Example() {
    return {.shader_hash = 0x123456789abcdef0ULL,
            .stage = 1,
            .logical_stage = 3,
            .sharp_index = 72,
            .array_index = 5,
            .descriptor_count = 16,
            .descriptor_stride_dw = 8,
            .reasons = 4,
            .is_r128 = false,
            .words = {0x10203040, 0x50607080, 0x90a0b0c0, 0xd0e0f001, 0x12345678, 0x87654321,
                      0xaabbccdd, 0xeeff1122}};
}

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    EXPECT_TRUE(input.is_open());
    std::string text{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    std::erase(text, '\r');
    return text;
}

TEST(TextureRejectionDiagnostics, RequiresExactExplicitOptIn) {
    EXPECT_TRUE(Vulkan::ParseTextureRejectionDiagnostics("1"));
    for (const auto value : {"", "0", "true", "yes", " 1", "1 ", "01", "+1"}) {
        EXPECT_FALSE(Vulkan::ParseTextureRejectionDiagnostics(value)) << value;
    }
}

TEST(TextureRejectionDiagnostics, DisabledDoesNotBuildOrRetainTuple) {
    TextureRejectionDiagnostics diagnostics{false};
    unsigned builds = 0;
    EXPECT_FALSE(diagnostics.Observe([&] {
        ++builds;
        return Example();
    }));
    EXPECT_EQ(builds, 0U);
    EXPECT_EQ(diagnostics.RecordedCount(), 0U);
}

TEST(TextureRejectionDiagnostics, EnabledRetainsEveryRawWordAndContextField) {
    TextureRejectionDiagnostics diagnostics{true};
    const auto expected = Example();
    const auto actual = diagnostics.Observe([&] { return expected; });
    ASSERT_TRUE(actual);
    EXPECT_EQ(*actual, expected);
    EXPECT_EQ(diagnostics.RecordedCount(), 1U);
}

TEST(TextureRejectionDiagnostics, DuplicateDoesNotConsumeBudget) {
    TextureRejectionDiagnostics diagnostics{true};
    ASSERT_TRUE(diagnostics.Observe(Example));
    for (unsigned repeat = 0; repeat < 100; ++repeat) {
        EXPECT_FALSE(diagnostics.Observe(Example));
    }
    EXPECT_EQ(diagnostics.RecordedCount(), 1U);
}

TEST(TextureRejectionDiagnostics, ChangedCapturedRootAndCountsRemainDistinct) {
    TextureRejectionDiagnostics diagnostics{true};
    std::array<std::uint32_t, 16> captured{};
    captured.fill(0x800000U);
    auto value = Example();
    value.roots = Vulkan::CaptureSrtRootProvenance(captured, 16U);
    const auto original = diagnostics.Observe([&] { return value; });
    ASSERT_TRUE(original);
    for (unsigned i = 0; i < captured.size(); ++i) {
        auto changed = captured;
        ++changed[i];
        value.roots = Vulkan::CaptureSrtRootProvenance(changed, 16U);
        EXPECT_TRUE(diagnostics.Observe([&] { return value; }));
    }
    value.roots = Vulkan::CaptureSrtRootProvenance(captured, 8U);
    EXPECT_TRUE(diagnostics.Observe([&] { return value; }));
    EXPECT_EQ(diagnostics.RecordedCount(), 18U);
    EXPECT_EQ(original->roots.words, captured);
}

TEST(TextureRejectionDiagnosticWiring, CapturedRootsComeFromOwnedFlatBuffer) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto start = source.find("const auto build_texture_observation =");
    ASSERT_NE(start, std::string::npos);
    const auto end = source.find("return observation;", start);
    ASSERT_NE(end, std::string::npos);
    const auto builder = source.substr(start, end - start);
    EXPECT_NE(builder.find("CaptureSrtRootProvenance(stage.flattened_ud_buf"), std::string::npos);
    EXPECT_EQ(builder.find("stage.user_data.data()"), std::string::npos);
    EXPECT_EQ(builder.find("liverpool->regs"), std::string::npos);
    EXPECT_NE(source.find("captured_ud=[", end), std::string::npos);
    EXPECT_NE(source.find("rejected->roots", end), std::string::npos);
}

TEST(TextureImageAdmission, UnsupportedImageIsNotPublishedToTheTextureCache) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_TEXTURE_CACHE_PATH);
    const auto insert = source.find("slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info)");
    ASSERT_NE(insert, std::string::npos);
    const auto validity = source.find("IsValid()", insert);
    const auto register_image = source.find("RegisterImage(image_id)", insert);
    ASSERT_NE(register_image, std::string::npos);
    EXPECT_NE(validity, std::string::npos);
    EXPECT_LT(validity, register_image);
    EXPECT_NE(source.find("slot_images.erase", validity), std::string::npos);
}

TEST(TextureImageAdmission, UndefinedFormatIsNotPublishedByInsertImage) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_TEXTURE_CACHE_PATH);
    const auto insert = source.find("ImageId TextureCache::InsertImage(");
    ASSERT_NE(insert, std::string::npos);
    const auto validity = source.find("!image.IsValid()", insert);
    const auto register_image = source.find("RegisterImage(image_id)", insert);
    ASSERT_NE(validity, std::string::npos);
    ASSERT_NE(register_image, std::string::npos);
    EXPECT_LT(validity, register_image);
    const auto check = source.substr(validity, register_image - validity);
    // The undefined-format stencil association path inserts directly and is not an InsertImage
    // caller.  Therefore this helper must not exempt vk::Format::eUndefined from validity
    // checking and registration.
    EXPECT_EQ(check.find("info.pixel_format != vk::Format::eUndefined"), std::string::npos);
}

TEST(TextureImageAdmission, UnsupportedImageConstructionFailsClosed) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_PATH);
    const auto unsupported = source.find("if (!selected_usage)");
    ASSERT_NE(unsupported, std::string::npos);
    const auto return_path = source.find("return;", unsupported);
    ASSERT_NE(return_path, std::string::npos);
    const auto backing_creation = source.find("backing = &backing_images.emplace_back()", unsupported);
    ASSERT_NE(backing_creation, std::string::npos);
    EXPECT_LT(return_path, backing_creation);
}

#if defined(SHADPS4_ENABLE_IMAGE_BACKING_CONTRACT_WIRING) &&                                  \
    defined(SHADPS4_TEXTURE_REJECTION_IMAGE_HEADER_PATH)

TEST(TextureImageAdmission, LowLevelImageOperationsCheckBackingBeforeDereference) {
    const auto header = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_HEADER_PATH);
    const auto image = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_PATH);

    struct AccessContract {
        const char* source;
        const char* function;
        const char* first_backing_access;
    };
    // This is deliberately opt-in while the low-level contract remains an open follow-up.  It
    // is a source-level RED gate: callers normally receive IDs only after TextureCache admission,
    // but an invalid Image object can still be constructed by the backing-less stencil and
    // allocation-failure paths.  Every listed operation has a safe no-op/false/empty result type;
    // FindView is excluded because its reference return type has no fail-closed sentinel.
    const std::array contracts{
        AccessContract{header.c_str(), "vk::Image GetImage() const", "backing->image.image"},
        AccessContract{header.c_str(), "vk::Extent3D GetHostExtent() const",
                       "backing->image.image_ci.extent"},
        AccessContract{image.c_str(), "bool Image::CanHostScale(", "backing->image"},
        AccessContract{image.c_str(), "bool Image::SetHostScale(", "backing->image"},
        AccessContract{image.c_str(), "Image::Barriers Image::GetBarriers(",
                       "auto& last_state = backing->"},
        AccessContract{image.c_str(), "void Image::CopyImage(", "src_image.GetHostExtent()"},
        AccessContract{image.c_str(), "void Image::CopyMip(", "ASSERT(SetHostScale(100))"},
        AccessContract{image.c_str(), "void Image::Resolve(", "SetBackingSamples(1, false)"},
        AccessContract{image.c_str(), "void Image::Clear(", "scheduler->EndRendering()"},
    };

    for (const auto& contract : contracts) {
        const auto function = std::string_view{contract.source}.find(contract.function);
        ASSERT_NE(function, std::string_view::npos) << contract.function;
        const auto access = std::string_view{contract.source}.find(contract.first_backing_access,
                                                                    function);
        ASSERT_NE(access, std::string_view::npos) << contract.function;
        const auto guard = std::string_view{contract.source}.find("IsValid()", function);
        ASSERT_NE(guard, std::string_view::npos) << contract.function;
        EXPECT_LT(guard, access) << contract.function;
    }
}

#endif

TEST(TextureImageAdmission, QueriedArrayLayerLimitIsCheckedBeforeImageCreation) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_PATH);
    const auto selected = source.find("supported_properties = properties.value");
    ASSERT_NE(selected, std::string::npos);
    const auto max_layers = source.find("imageFormatProperties.maxArrayLayers", selected);
    const auto image_create = source.find("const vk::ImageCreateInfo image_ci", selected);
    const auto backing_creation = source.find("backing = &backing_images.emplace_back()", selected);
    ASSERT_NE(max_layers, std::string::npos);
    ASSERT_NE(image_create, std::string::npos);
    ASSERT_NE(backing_creation, std::string::npos);
    // A malformed array count must be rejected using the device's queried limit before the
    // VkImageCreateInfo or backing image is published.  Otherwise Vulkan validation can observe
    // an impossible image and a later tile copy can address every bogus layer.
    EXPECT_LT(max_layers, image_create);
    EXPECT_LT(max_layers, backing_creation);
}

TEST(TextureImageAdmission, HostArrayLayerAdmissionIsFailClosedAtTheDeviceLimit) {
    EXPECT_FALSE(VideoCore::IsHostImageArrayLayerCountSupported(0, 2048));
    EXPECT_FALSE(VideoCore::IsHostImageArrayLayerCountSupported(1, 0));
    EXPECT_TRUE(VideoCore::IsHostImageArrayLayerCountSupported(1, 2048));
    EXPECT_TRUE(VideoCore::IsHostImageArrayLayerCountSupported(2048, 2048));
    EXPECT_FALSE(VideoCore::IsHostImageArrayLayerCountSupported(2049, 2048));
}

TEST(TextureImageAdmission, HostGeometryMustFitQueriedFormatLimits) {
    constexpr VideoCore::Extent3D extent{1920, 1080, 1};
    EXPECT_TRUE(VideoCore::IsHostImageGeometrySupported(extent, 11, {4096, 4096, 1}, 12));
    EXPECT_FALSE(VideoCore::IsHostImageGeometrySupported(extent, 13, {4096, 4096, 1}, 12));
    EXPECT_FALSE(VideoCore::IsHostImageGeometrySupported({4097, 1080, 1}, 1,
                                                          {4096, 4096, 1}, 12));
    EXPECT_FALSE(VideoCore::IsHostImageGeometrySupported({1920, 1081, 1}, 1,
                                                          {4096, 1080, 1}, 12));
    EXPECT_FALSE(VideoCore::IsHostImageGeometrySupported({1, 1, 2}, 1, {4096, 4096, 1}, 12));
    EXPECT_FALSE(VideoCore::IsHostImageGeometrySupported(extent, 1, {0, 0, 0}, 0));
}

TEST(TextureImageAdmission, HostGeometryIsCheckedBeforeImageCreation) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_PATH);
    const auto selected = source.find("supported_properties = properties.value");
    ASSERT_NE(selected, std::string::npos);
    const auto geometry = source.find("IsHostImageGeometrySupported(", selected);
    const auto image_create = source.find("const vk::ImageCreateInfo image_ci", selected);
    const auto backing_creation = source.find("backing = &backing_images.emplace_back()", selected);
    ASSERT_NE(geometry, std::string::npos);
    ASSERT_NE(image_create, std::string::npos);
    ASSERT_NE(backing_creation, std::string::npos);
    EXPECT_LT(geometry, image_create);
    EXPECT_LT(geometry, backing_creation);
}

TEST(TextureImageAdmission, BankSwizzleRejectsInvalidCalculatedMacroMode) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RESOURCE_PATH);
    const auto bank_swizzle = source.find("u8 GetBankSwizzle() const");
    ASSERT_NE(bank_swizzle, std::string::npos);
    const auto calculated = source.find("CalculateMacrotileMode", bank_swizzle);
    const auto lookup = source.find("GetAltNumBanks(macro_tile_mode)", bank_swizzle);
    ASSERT_NE(calculated, std::string::npos);
    ASSERT_NE(lookup, std::string::npos);
    // Guest BC/packed descriptors can produce a sub-64-byte tile and therefore an invalid
    // calculated enum.  Validate that result before dispatching to the exhaustive bank table;
    // otherwise malformed descriptors reach UNREACHABLE in GetAltNumBanks.
    const auto validity = source.find("IsValidMacroTileMode", calculated);
    ASSERT_NE(validity, std::string::npos);
    EXPECT_LT(validity, lookup);
}

TEST(TextureImageAdmission, MacroTileModeValidityBoundsGuestDerivedLookup) {
    EXPECT_TRUE(AmdGpu::IsValidMacroTileMode(AmdGpu::MacroTileMode::Mode_1x1_2_Dup3));
    EXPECT_FALSE(AmdGpu::IsValidMacroTileMode(
        static_cast<AmdGpu::MacroTileMode>(std::numeric_limits<std::uint32_t>::max())));

    auto image = AmdGpu::Image::Null(false);
    image.alt_tile_mode = 1;
    image.tiling_index = static_cast<std::uint64_t>(AmdGpu::TileMode::ThinThinPrt);
    image.data_format = static_cast<std::uint64_t>(AmdGpu::DataFormat::FormatBc1);
    // A 4-bit element yields a 32-byte micro-tile.  The calculated macro mode underflows before
    // the guard and must therefore be treated as invalid rather than indexing GetAltNumBanks.
    EXPECT_EQ(image.GetBankSwizzle(), 0U);
}

TEST(TextureImageAdmission, FindImageRejectsViewRangesOutsideGuestResources) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_TEXTURE_CACHE_PATH);
    const auto find = source.find("ImageId TextureCache::FindImage(");
    ASSERT_NE(find, std::string::npos);
    const auto range_guard =
        source.find("IsValidSubresourceRange(desc.view_info.range, info.resources)", find);
    const auto cache_scan = source.find("ForEachImageInRegion", find);
    ASSERT_NE(cache_scan, std::string::npos);
    // Admission must happen before an invalid view range can participate in cache overlap
    // resolution or image creation.
    EXPECT_NE(range_guard, std::string::npos);
    EXPECT_LT(range_guard, cache_scan);
}

TEST(TextureImageAdmission, FindImageFromRangeSkipsBackinglessPlaceholders) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_TEXTURE_CACHE_PATH);
    const auto find = source.find("ImageId TextureCache::FindImageFromRange(");
    ASSERT_NE(find, std::string::npos);
    const auto callback = source.find("ForEachImageInRegion(address, size", find);
    const auto guest_address = source.find("image.info.guest_address", callback);
    ASSERT_NE(callback, std::string::npos);
    ASSERT_NE(guest_address, std::string::npos);
    // Stencil association entries are intentionally backing-less.  They may still be present in
    // the address index, but a range lookup must not return one to copy/clear paths that dereference
    // the host image even when ensure_valid is false.
    const auto validity = source.find("image.IsValid()", callback);
    ASSERT_NE(validity, std::string::npos);
    EXPECT_LT(validity, guest_address);
}

TEST(TextureImageAdmission, FindImageSkipsBackinglessCandidatesBeforeOverlap) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_TEXTURE_CACHE_PATH);
    const auto find = source.find("ImageId TextureCache::FindImage(ImageDesc& desc");
    ASSERT_NE(find, std::string::npos);
    const auto callback = source.find("ForEachImageInRegion(info.guest_address, info.guest_size",
                                     find);
    const auto publish = source.find("image_ids.push_back(image_id)", callback);
    ASSERT_NE(callback, std::string::npos);
    ASSERT_NE(publish, std::string::npos);
    // Stencil association records are registered in the same page index as normal images but do
    // not own a Vulkan image.  Do not let one enter overlap resolution, where a cache hit can be
    // returned and subsequently dereferenced as a real image.
    const auto validity = source.find("image.IsValid()", callback);
    ASSERT_NE(validity, std::string::npos);
    EXPECT_LT(validity, publish);
}

TEST(TextureImageAdmission, FinalDescriptorResolutionRejectsStaleViewRanges) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto finalize = source.find("bool Rasterizer::FinalizeImageDescriptors(");
    ASSERT_NE(finalize, std::string::npos);
    const auto view = source.find("FindView(pending.view_info, false)", finalize);
    ASSERT_NE(view, std::string::npos);
    const auto range_guard = source.find("IsValidSubresourceRange(pending.view_info.range,",
                                        finalize);
    const auto range_resource = source.find("image->info.resources", range_guard);
    EXPECT_NE(range_guard, std::string::npos);
    EXPECT_NE(range_resource, std::string::npos);
    EXPECT_LT(range_guard, view);
    EXPECT_LT(range_resource, view);
}

TEST(TextureImageAdmission, FindImageRejectsMipChainsLongerThanTheImageExtent) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_TEXTURE_CACHE_PATH);
    const auto find = source.find("ImageId TextureCache::FindImage(");
    ASSERT_NE(find, std::string::npos);
    const auto mip_guard = source.find("HasValidMipChain()", find);
    const auto cache_scan = source.find("ForEachImageInRegion", find);
    ASSERT_NE(cache_scan, std::string::npos);
    EXPECT_NE(mip_guard, std::string::npos);
    EXPECT_LT(mip_guard, cache_scan);
}

TEST(TextureImageAdmission, InsertImageRejectsMipChainsBeforeSlotConstruction) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_TEXTURE_CACHE_PATH);
    const auto insert = source.find("ImageId TextureCache::InsertImage(");
    ASSERT_NE(insert, std::string::npos);
    const auto mip_guard = source.find("HasValidMipChain()", insert);
    const auto slot_insert = source.find("slot_images.insert", insert);
    ASSERT_NE(slot_insert, std::string::npos);
    EXPECT_NE(mip_guard, std::string::npos);
    EXPECT_LT(mip_guard, slot_insert);
}

TEST(TextureImageAdmission, MipChainBoundMatchesVulkanExtent) {
    VideoCore::ImageInfo info{};
    info.size = {1, 1, 1};
    info.resources = {1, 1};
    info.guest_size = 1;
    info.mips_layout[0] = {1, 1, 1, 0};
    EXPECT_TRUE(info.HasValidMipChain());
    info.resources.levels = 2;
    info.guest_size = 2;
    info.mips_layout[1] = {1, 1, 1, 1};
    EXPECT_FALSE(info.HasValidMipChain());

    info.size = {260, 1, 1};
    info.resources.levels = 9;
    info.guest_size = 9;
    for (u32 mip = 2; mip < 9; ++mip) {
        info.mips_layout[mip] = {1, 1, 1, mip};
    }
    EXPECT_TRUE(info.HasValidMipChain());
    info.resources.levels = 10;
    info.guest_size = 10;
    info.mips_layout[9] = {1, 1, 1, 9};
    EXPECT_FALSE(info.HasValidMipChain());

    info.props.is_volume = 1;
    info.size = {1, 1, 512};
    info.resources.levels = 10;
    info.guest_size = 10;
    for (u32 mip = 0; mip < 10; ++mip) {
        info.mips_layout[mip] = {1, 1, 1, mip};
    }
    EXPECT_TRUE(info.HasValidMipChain());
    info.resources.levels = 11;
    info.guest_size = 11;
    info.mips_layout[10] = {1, 1, 1, 10};
    EXPECT_FALSE(info.HasValidMipChain());
}

TEST(TextureImageAdmission, MipChainRequiresARepresentablePublishedLayout) {
    VideoCore::ImageInfo info{};
    info.size = {1, 1, 1};
    info.resources = {1, 1};
    info.tile_mode = AmdGpu::TileMode::DisplayLinearGeneral;
    info.array_mode = AmdGpu::ArrayMode::ArrayLinearGeneral;

    // A default/partially-built ImageInfo must not be admitted merely because its extent and
    // resource counts look valid.  InsertImage would otherwise publish a host image whose guest
    // footprint and mip table contain no bytes to upload or track.
    EXPECT_FALSE(info.HasValidMipChain());

    info.guest_size = 1;
    info.mips_layout[0] = {1, 1, 1, 0};
    EXPECT_TRUE(info.HasValidMipChain());

    info.mips_layout[0].size = 0;
    EXPECT_FALSE(info.HasValidMipChain());
    info.mips_layout[0] = {1, 1, 1, 1};
    EXPECT_FALSE(info.HasValidMipChain());
}

TEST(TextureImageAdmission, TiledSampleCountIsBoundedBeforeLayoutDerivation) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_INFO_PATH);
    const auto update = source.find("void ImageInfo::UpdateSize() {");
    ASSERT_NE(update, std::string::npos);
    const auto guard = source.find("num_samples > 8", update);
    const auto layout = source.find("ClassifyImageArrayLayout", update);
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(layout, std::string::npos);
    // GetMacroTileExtents currently has geometry tables only through x8. A 16-sample tiled
    // descriptor must be rejected before layout selection rather than indexing past that table.
    EXPECT_LT(guard, layout);
}

TEST(TextureImageAdmission, MsaaSampleExponentIsBoundedBeforeImageCreation) {
    auto image = AmdGpu::Image::Null(false);
    for (const auto type : {AmdGpu::ImageType::Color2DMsaa,
                            AmdGpu::ImageType::Color2DMsaaArray}) {
        image.type = static_cast<std::uint64_t>(type);
        for (unsigned exponent = 0; exponent <= 4; ++exponent) {
            image.last_level = exponent;
            EXPECT_TRUE(image.HasValidMsaaSampleExponent());
        }
        for (const unsigned exponent : {5U, 6U, 15U}) {
            image.last_level = exponent;
            EXPECT_FALSE(image.HasValidMsaaSampleExponent());
        }
    }
    image.type = static_cast<std::uint64_t>(AmdGpu::ImageType::Color2D);
    image.last_level = 15;
    EXPECT_TRUE(image.HasValidMsaaSampleExponent());
}

TEST(TextureImageAdmission, MsaaDescriptorIsRejectedBeforeTextureCacheLookup) {
    const auto source = ReadSource(SHADPS4_RESOURCE_TRACKING_SOURCE_PATH);
    const auto patch = source.find("void PatchImageSharp(");
    ASSERT_NE(patch, std::string::npos);
    const auto descriptor = source.find("auto image = image_res.GetSharp(info)", patch);
    ASSERT_NE(descriptor, std::string::npos);
    const auto guard = source.find("HasValidMsaaSampleExponent", descriptor);
    ASSERT_NE(guard, std::string::npos);
    const auto fail = source.find("info.translation_failed = true", guard);
    ASSERT_NE(fail, std::string::npos);
    EXPECT_LT(guard, fail);
}

TEST(TextureImageAdmission, DescriptorRangeRejectionLogsIndependentValidityDimensions) {
    const auto source = ReadSource(SHADPS4_RESOURCE_TRACKING_SOURCE_PATH);
    const auto patch = source.find("void PatchImageSharp(");
    ASSERT_NE(patch, std::string::npos);
    const auto descriptor = source.find("auto image = image_res.GetSharp(info)", patch);
    ASSERT_NE(descriptor, std::string::npos);

    // A combined HasValidSubresourceRange() result cannot identify whether a malformed
    // descriptor failed its MSAA exponent, mip window, or array window. Keep all three
    // dimensions observable in the rejection diagnostic before translation is failed closed.
    const auto msaa = source.find("const bool valid_msaa = image.HasValidMsaaSampleExponent();",
                                 descriptor);
    const auto mip = source.find("const bool valid_mip = image.HasValidMipRange();", descriptor);
    const auto array = source.find("const bool valid_array = image.HasValidArrayRange();", descriptor);
    const auto rejection = source.find("rejecting shader", descriptor);
    const auto fail = source.find("info.translation_failed = true", descriptor);
    const auto logged_msaa = source.find("valid_msaa={}", descriptor);
    const auto logged_mip = source.find("valid_mip={}", descriptor);
    const auto logged_array = source.find("valid_array={}", descriptor);
    ASSERT_NE(msaa, std::string::npos);
    ASSERT_NE(mip, std::string::npos);
    ASSERT_NE(array, std::string::npos);
    ASSERT_NE(rejection, std::string::npos);
    ASSERT_NE(fail, std::string::npos);
    ASSERT_NE(logged_msaa, std::string::npos);
    ASSERT_NE(logged_mip, std::string::npos);
    ASSERT_NE(logged_array, std::string::npos);
    EXPECT_LT(msaa, rejection);
    EXPECT_LT(mip, rejection);
    EXPECT_LT(array, rejection);
    EXPECT_LT(rejection, fail);
    EXPECT_LT(logged_msaa, rejection);
    EXPECT_LT(logged_mip, rejection);
    EXPECT_LT(logged_array, rejection);
}

TEST(TextureImageAdmission, InvertedMipWindowIsRejectedBeforeDescriptorPublication) {
    const auto source = ReadSource(SHADPS4_RESOURCE_TRACKING_SOURCE_PATH);
    const auto patch = source.find("void PatchImageSharp(");
    ASSERT_NE(patch, std::string::npos);
    const auto descriptor = source.find("auto image = image_res.GetSharp(info)", patch);
    ASSERT_NE(descriptor, std::string::npos);
    // Array-window admission now shares the same guard as the mip window.  Keep this contract
    // tolerant of the stronger combined predicate while still requiring it before publication.
    const auto range_guard = source.find("HasValidSubresourceRange", descriptor);
    const auto descriptor_add = source.find("descriptors.Add(image_res)", descriptor);
    ASSERT_NE(range_guard, std::string::npos);
    ASSERT_NE(descriptor_add, std::string::npos);
    EXPECT_LT(range_guard, descriptor_add);
    const auto failure = source.find("info.translation_failed = true", range_guard);
    ASSERT_NE(failure, std::string::npos);
    EXPECT_LT(failure, descriptor_add);
}

TEST(TextureImageAdmission, RuntimeZeroImageBindingUsesNullDescriptor) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto begin = source.find("void Rasterizer::BindTextures(");
    ASSERT_NE(begin, std::string::npos);
    const auto zero_guard = source.find("if (num_bindings == 0)", begin);
    const auto invalid = source.find("append_invalid_image_binding", zero_guard);
    const auto descriptor_array_size = source.find("image_descriptor_array_sizes.push_back(1)",
                                                    zero_guard);
    ASSERT_NE(zero_guard, std::string::npos);
    ASSERT_NE(invalid, std::string::npos);
    ASSERT_NE(descriptor_array_size, std::string::npos);
    EXPECT_LT(zero_guard, invalid);
    EXPECT_LT(invalid, descriptor_array_size);
}

TEST(TextureImageAdmission, InvalidArrayWindowIsRejectedBeforeImageViewConstruction) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto begin = source.find("void Rasterizer::BindTextures(");
    ASSERT_NE(begin, std::string::npos);
    const auto sharp = source.find("const auto tsharp = image_desc.GetSharp(stage, i);", begin);
    const auto image_desc = source.find("std::tuple{tsharp, image_desc}", sharp);
    ASSERT_NE(sharp, std::string::npos);
    ASSERT_NE(image_desc, std::string::npos);

    // Array and cube descriptors carry an inclusive guest layer window.  This check must happen
    // after reading the raw T# but before ImageDesc constructs ImageInfo/ImageViewInfo; otherwise
    // ImageViewInfo's defensive clamp can silently redirect an invalid base_array to another
    // layer and let a malformed view reach Vulkan.
    const auto range_guard = source.find("HasValidSubresourceRange()", sharp);
    ASSERT_NE(range_guard, std::string::npos);
    EXPECT_LT(range_guard, image_desc);
}

TEST(TextureImageAdmission, ArrayAndCubeWindowsRejectOutOfBoundsLayers) {
    auto array = AmdGpu::Image::Null(false);
    array.type = static_cast<std::uint64_t>(AmdGpu::ImageType::Color2DArray);
    array.depth = 3;
    array.base_array = 1;
    array.last_array = 3;
    EXPECT_TRUE(array.HasValidSubresourceRange());

    array.base_array = 3;
    array.last_array = 4;
    EXPECT_FALSE(array.HasValidSubresourceRange());

    auto cube = AmdGpu::Image::Null(false);
    cube.type = static_cast<std::uint64_t>(AmdGpu::ImageType::Cube);
    cube.depth = 0; // One cube contains six array layers.
    cube.base_array = 0;
    cube.last_array = 5;
    EXPECT_TRUE(cube.HasValidSubresourceRange());

    cube.base_array = 6;
    cube.last_array = 6;
    EXPECT_FALSE(cube.HasValidSubresourceRange());
}

TEST(TextureImageAdmission, BindTexturesRebindChecksImageIdBeforeUse) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto second_pass = source.find("// Second pass to re-bind images");
    ASSERT_NE(second_pass, std::string::npos);
    const auto find = source.find("image_id = texture_cache.FindImage(desc);", second_pass);
    ASSERT_NE(find, std::string::npos);
    const auto guard = source.find("if (!image_id)", find);
    const auto track = source.find("TrackBoundImage(image_id);", find);
    const auto get = source.find("GetImage(image_id)", find);
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(track, std::string::npos);
    ASSERT_NE(get, std::string::npos);
    EXPECT_LT(guard, track);
    EXPECT_LT(guard, get);
}

TEST(TextureImageAdmission, BeginRenderingRebindChecksImageIdBeforeUse) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto begin = source.find("Rasterizer::BeginRendering(");
    ASSERT_NE(begin, std::string::npos);
    const auto find = source.find("image_id = texture_cache.FindImage(desc);", begin);
    ASSERT_NE(find, std::string::npos);
    const auto guard = source.find("if (!image_id)", find);
    const auto track = source.find("TrackBoundImage(image_id);", find);
    const auto get = source.find("GetImage(image_id)", find);
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(track, std::string::npos);
    ASSERT_NE(get, std::string::npos);
    EXPECT_LT(guard, track);
    EXPECT_LT(guard, get);
}

TEST(TextureImageAdmission, ResolveChecksBothImageIdsBeforeGetImage) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto resolve = source.find("void Rasterizer::Resolve()");
    ASSERT_NE(resolve, std::string::npos);
    const auto find0 = source.find("const auto mrt0_id = texture_cache.FindImage(mrt0_desc, true);",
                                   resolve);
    const auto find1 = source.find("const auto mrt1_id = texture_cache.FindImage(mrt1_desc, true);",
                                   resolve);
    ASSERT_NE(find0, std::string::npos);
    ASSERT_NE(find1, std::string::npos);
    const auto guard = source.find("if (!mrt0_id || !mrt1_id)", find1);
    const auto get0 = source.find("GetImage(mrt0_id)", find1);
    const auto get1 = source.find("GetImage(mrt1_id)", find1);
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(get0, std::string::npos);
    ASSERT_NE(get1, std::string::npos);
    EXPECT_LT(guard, get0);
    EXPECT_LT(guard, get1);
    EXPECT_EQ(source.find("GetImage(texture_cache.FindImage(mrt0_desc, true))", resolve),
              std::string::npos);
    EXPECT_EQ(source.find("GetImage(texture_cache.FindImage(mrt1_desc, true))", resolve),
              std::string::npos);
}

TEST(TextureImageAdmission, DepthStencilCopyChecksBothImageIdsBeforeGetImage) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto copy = source.find("void Rasterizer::DepthStencilCopy(");
    ASSERT_NE(copy, std::string::npos);
    const auto find0 = source.find("const auto read_image_id = texture_cache.FindImage(read_desc);",
                                   copy);
    const auto find1 = source.find("const auto write_image_id = texture_cache.FindImage(write_desc);",
                                   copy);
    ASSERT_NE(find0, std::string::npos);
    ASSERT_NE(find1, std::string::npos);
    const auto guard = source.find("if (!read_image_id || !write_image_id)", find1);
    const auto get0 = source.find("GetImage(read_image_id)", find1);
    const auto get1 = source.find("GetImage(write_image_id)", find1);
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(get0, std::string::npos);
    ASSERT_NE(get1, std::string::npos);
    EXPECT_LT(guard, get0);
    EXPECT_LT(guard, get1);
    EXPECT_EQ(source.find("GetImage(texture_cache.FindImage(read_desc))", copy),
              std::string::npos);
    EXPECT_EQ(source.find("GetImage(texture_cache.FindImage(write_desc))", copy),
              std::string::npos);
}

TEST(TextureImageAdmission, ResolveRejectsInvalidRangesBeforeTransfer) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_PATH);
    const auto resolve = source.find("void Image::Resolve(");
    ASSERT_NE(resolve, std::string::npos);
    const auto source_guard =
        source.find("IsValidSubresourceRange(mrt0_range, src_image.info.resources)", resolve);
    const auto destination_guard =
        source.find("IsValidSubresourceRange(mrt1_range, info.resources)", resolve);
    const auto source_transit = source.find("src_image.Transit(", resolve);
    const auto destination_transit = source.find("Transit(", source_transit + 1);
    const auto copy = source.find("copyImage(", resolve);
    ASSERT_NE(source_guard, std::string::npos);
    ASSERT_NE(destination_guard, std::string::npos);
    ASSERT_NE(source_transit, std::string::npos);
    ASSERT_NE(destination_transit, std::string::npos);
    ASSERT_NE(copy, std::string::npos);
    EXPECT_LT(source_guard, source_transit);
    EXPECT_LT(destination_guard, source_transit);
    EXPECT_LT(source_guard, copy);
    EXPECT_LT(destination_guard, copy);
}

TEST(TextureImageAdmission, ClearRejectsInvalidRangeBeforeTransfer) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_PATH);
    const auto clear = source.find("void Image::Clear(");
    ASSERT_NE(clear, std::string::npos);
    const auto guard = source.find("IsValidSubresourceRange(range, info.resources)", clear);
    const auto end_rendering = source.find("scheduler->EndRendering();", clear);
    const auto clear_command = source.find("clearColorImage(", clear);
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(end_rendering, std::string::npos);
    ASSERT_NE(clear_command, std::string::npos);
    EXPECT_LT(guard, end_rendering);
    EXPECT_LT(guard, clear_command);
}

TEST(TextureImageAdmission, DepthStencilCopyRejectsInvalidRangeBeforeTransfer) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto copy = source.find("void Rasterizer::DepthStencilCopy(");
    ASSERT_NE(copy, std::string::npos);
    const auto range = source.find("sub_range.extent.layers", copy);
    const auto guard = source.find("IsValidSubresourceRange(sub_range,", copy);
    const auto source_transit = source.find("read_image.Transit(", copy);
    const auto copy_command = source.find("copyImage(", copy);
    ASSERT_NE(range, std::string::npos);
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(source_transit, std::string::npos);
    ASSERT_NE(copy_command, std::string::npos);
    EXPECT_LT(range, guard);
    EXPECT_LT(guard, source_transit);
    EXPECT_LT(guard, copy_command);
}

TEST(TextureImageAdmission, FastClearEliminationChecksImageAdmissionBeforeUse) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto eliminate = source.find("void Rasterizer::EliminateFastClear()");
    ASSERT_NE(eliminate, std::string::npos);
    const auto find = source.find("const auto image_id = texture_cache.FindImage(desc);", eliminate);
    const auto guard = source.find("if (!image_id)", find);
    const auto view = source.find("FindRenderTarget(image_id, desc)", find);
    const auto clear = source.find("image.Clear(clear_value, desc.view_info.range)", find);
    ASSERT_NE(find, std::string::npos);
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(view, std::string::npos);
    ASSERT_NE(clear, std::string::npos);
    EXPECT_LT(find, guard);
    EXPECT_LT(guard, view);
    EXPECT_LT(guard, clear);
}

TEST(TextureRejectionDiagnostics, DistinctRawWordsAreNotDeduplicatedAway) {
    TextureRejectionDiagnostics diagnostics{true};
    ASSERT_TRUE(diagnostics.Observe(Example));
    for (unsigned word = 0; word < 8; ++word) {
        auto value = Example();
        value.words[word] ^= 1U;
        EXPECT_TRUE(diagnostics.Observe([&] { return value; }));
    }
    EXPECT_EQ(diagnostics.RecordedCount(), 9U);
}

TEST(TextureRejectionDiagnostics, DistinguishesStageShaderSharpArrayAndReason) {
    TextureRejectionDiagnostics diagnostics{true};
    ASSERT_TRUE(diagnostics.Observe(Example));
    for (unsigned field = 0; field < 10; ++field) {
        auto value = Example();
        switch (field) {
        case 0:
            ++value.shader_hash;
            break;
        case 1:
            ++value.stage;
            break;
        case 2:
            ++value.logical_stage;
            break;
        case 3:
            ++value.sharp_index;
            break;
        case 4:
            ++value.array_index;
            break;
        case 5:
            ++value.descriptor_count;
            break;
        case 6:
            ++value.descriptor_stride_dw;
            break;
        case 7:
            ++value.reasons;
            break;
        case 8:
            value.is_r128 = true;
            break;
        case 9:
            value.base_byte_logical_gpu_mapped = true;
            break;
        }
        EXPECT_TRUE(diagnostics.Observe([&] { return value; }));
    }
    EXPECT_EQ(diagnostics.RecordedCount(), 11U);
}

TEST(TextureRejectionDiagnostics, OwnsTupleAcrossSourceMutation) {
    TextureRejectionDiagnostics diagnostics{true};
    auto source = Example();
    const auto first = diagnostics.Observe([&] { return source; });
    ASSERT_TRUE(first);
    source.words.fill(0);
    EXPECT_EQ(*first, Example());
    EXPECT_FALSE(diagnostics.Observe(Example));
    EXPECT_TRUE(diagnostics.Observe([&] { return source; }));
}

TEST(TextureRejectionDiagnostics, BudgetExhaustionDoesNotEvaluateTupleBuilder) {
    TextureRejectionDiagnostics diagnostics{true};
    for (unsigned index = 0; index < TextureRejectionDiagnostics::MaxObservations; ++index) {
        auto value = Example();
        value.array_index = index;
        ASSERT_TRUE(diagnostics.Observe([&] { return value; }));
    }
    unsigned builds = 0;
    EXPECT_FALSE(diagnostics.Observe([&] {
        ++builds;
        return Example();
    }));
    EXPECT_EQ(builds, 0U);
    EXPECT_EQ(diagnostics.RecordedCount(), TextureRejectionDiagnostics::MaxObservations);
}

TEST(TextureRejectionDiagnostics, ReasonMaskPreservesAddressAndFormatGuardIdentities) {
    for (const unsigned bits : {0U, 1U, 4U, 5U, 8U, 9U, 12U, 13U}) {
        const bool address_40bit = !(bits & 1);
        const bool valid_data = !(bits & 4);
        const bool valid_number = !(bits & 8);
        const auto reasons =
            Vulkan::TextureRejectionReasons(address_40bit, valid_data, valid_number);
        EXPECT_EQ(reasons, bits);
        EXPECT_EQ(reasons != 0, !address_40bit || !valid_data || !valid_number);
    }
}

constexpr std::array<std::uint32_t, 4> Selectors(AmdGpu::CompMapping mapping) {
    return {static_cast<std::uint32_t>(mapping.r), static_cast<std::uint32_t>(mapping.g),
            static_cast<std::uint32_t>(mapping.b), static_cast<std::uint32_t>(mapping.a)};
}

AmdGpu::Image ImageWithSelectors(std::array<std::uint32_t, 4> selectors) {
    auto image = AmdGpu::Image::Null(false);
    image.base_address = 0x10000;
    image.dst_sel_x = selectors[0];
    image.dst_sel_y = selectors[1];
    image.dst_sel_z = selectors[2];
    image.dst_sel_w = selectors[3];
    return image;
}

std::uint32_t SwizzleAdmission(const AmdGpu::Image& image, bool is_storage) {
    return Vulkan::TextureRejectionReasons(
        true, true, true,
        is_storage ||
            Vulkan::TextureComponentSelectorsValid({static_cast<std::uint32_t>(image.dst_sel_x),
                                                    static_cast<std::uint32_t>(image.dst_sel_y),
                                                    static_cast<std::uint32_t>(image.dst_sel_z),
                                                    static_cast<std::uint32_t>(image.dst_sel_w)}));
}

class ReservedTextureSelector : public testing::TestWithParam<std::tuple<unsigned, unsigned>> {};

TEST_P(ReservedTextureSelector, SampledDescriptorRefusedBeforeComponentConversion) {
    const auto [channel, reserved] = GetParam();
    std::array<std::uint32_t, 4> selectors{4, 5, 6, 7};
    selectors[channel] = reserved;
    const auto image = ImageWithSelectors(selectors);
    EXPECT_FALSE(Vulkan::TextureComponentSelectorsValid(selectors));
    EXPECT_EQ(SwizzleAdmission(image, false), 0x10U);
    // A storage view never consumes DstSelect; unused bits do not invalidate its binding.
    EXPECT_EQ(SwizzleAdmission(image, true), 0U);
}

INSTANTIATE_TEST_SUITE_P(EveryChannel, ReservedTextureSelector,
                         testing::Combine(testing::Values(0U, 1U, 2U, 3U),
                                          testing::Values(2U, 3U)));

TEST(TextureSwizzleAdmission, ValidSelectorsAndConstantsSurviveEveryRawFormatRemap) {
    constexpr std::array<unsigned, 6> valid{0, 1, 4, 5, 6, 7};
    // Exercise actual descriptor DstSelect/RemapSwizzle, including every format-specific
    // permutation and single-channel Alpha->Red rewrite. Format admission is separate.
    for (unsigned format = 0; format < 64; ++format) {
        for (const auto r : valid) {
            for (const auto g : valid) {
                for (const auto b : valid) {
                    for (const auto a : valid) {
                        auto image = ImageWithSelectors({r, g, b, a});
                        image.data_format = format;
                        const auto remapped = Selectors(image.DstSelect());
                        for (const auto selector : remapped) {
                            ASSERT_NE(std::ranges::find(valid, selector), valid.end());
                        }
                        ASSERT_TRUE(Vulkan::TextureComponentSelectorsValid(remapped));
                        ASSERT_EQ(SwizzleAdmission(image, false), 0U);
                    }
                }
            }
        }
    }
}

TEST(TextureSwizzleAdmission, NoFormatRemapMakesReservedSelectorsValid) {
    for (unsigned format = 0; format < 64; ++format) {
        for (unsigned channel = 0; channel < 4; ++channel) {
            for (const unsigned reserved : {2U, 3U}) {
                std::array<std::uint32_t, 4> selectors{4, 5, 6, 7};
                selectors[channel] = reserved;
                auto image = ImageWithSelectors(selectors);
                image.data_format = format;
                const auto remapped = Selectors(image.DstSelect());
                ASSERT_NE(std::ranges::find(remapped, reserved), remapped.end());
                ASSERT_FALSE(Vulkan::TextureComponentSelectorsValid(remapped));
            }
        }
    }
}

TEST(TextureSwizzleAdmission, ComponentReasonUsesNewBitAndRetainsHistoricalBits) {
    for (const unsigned bits : {0U, 1U, 4U, 5U, 8U, 9U, 12U, 13U}) {
        EXPECT_EQ(Vulkan::TextureRejectionReasons(!(bits & 1), !(bits & 4), !(bits & 8), false),
                  bits | 0x10U);
        EXPECT_EQ(Vulkan::TextureRejectionReasons(!(bits & 1), !(bits & 4), !(bits & 8), true),
                  bits);
    }
}

TEST(TextureSwizzleAdmission, RejectionRetainsOriginalDescriptorAndCapturedRoots) {
    TextureRejectionDiagnostics diagnostics{true};
    auto image = ImageWithSelectors({4, 2, 6, 3});
    std::array<std::uint32_t, 16> roots{};
    roots[0] = 0x12345678;
    auto expected = Example();
    expected.words = std::bit_cast<std::array<std::uint32_t, 8>>(image);
    expected.roots = Vulkan::CaptureSrtRootProvenance(roots, 16);
    expected.reasons = 0x10U;
    const auto observed = diagnostics.Observe([&] {
        auto observation = expected;
        observation.reasons = SwizzleAdmission(image, false);
        return observation;
    });
    ASSERT_TRUE(observed);
    image.dst_sel_y = 5;
    roots.fill(0);
    EXPECT_EQ(*observed, expected);
    EXPECT_FALSE(diagnostics.Observe([&] { return expected; }));
}

TEST(TextureSwizzleAdmission, SampledOnlyRawCheckPrecedesViewConstructorAndUsesNullBinding) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto begin = source.find("void Rasterizer::BindTextures(");
    ASSERT_NE(begin, std::string::npos);
    const auto reasons = source.find("TextureRejectionReasons(", begin);
    ASSERT_NE(reasons, std::string::npos);
    const auto reasons_end = source.find(';', reasons);
    ASSERT_NE(reasons_end, std::string::npos);
    auto call = source.substr(reasons, reasons_end - reasons);
    std::erase_if(call, [](unsigned char c) { return std::isspace(c); });
    EXPECT_NE(call.find("image_desc.is_written||TextureComponentSelectorsValid("),
              std::string::npos);
    for (const auto* field :
         {"tsharp.dst_sel_x", "tsharp.dst_sel_y", "tsharp.dst_sel_z", "tsharp.dst_sel_w"}) {
        EXPECT_NE(call.find(field), std::string::npos) << field;
    }
    const auto reject = source.find("if (rejection_reasons != 0)", reasons_end);
    ASSERT_NE(reject, std::string::npos);
    const auto null_binding =
        source.find("append_invalid_image_binding(image_desc.is_written);", reject);
    ASSERT_NE(null_binding, std::string::npos);
    const auto constructor = source.find("std::tuple{tsharp, image_desc}", reasons_end);
    ASSERT_NE(constructor, std::string::npos);
    EXPECT_LT(reject, null_binding);
    EXPECT_LT(null_binding, source.find("continue;", null_binding));
    EXPECT_LT(source.find("continue;", null_binding), constructor);
}

// Exercise the same block conversion, power-of-two padding, and tile sizing used by
// ImageInfo::UpdateSize. Raw pitch is in texels; it is not the final padded row extent.
void ExpectPaddedImageAdmission(unsigned pitch, unsigned width, unsigned bits_per_block,
                                bool block_compressed, bool pow2) {
    constexpr unsigned height = 64;
    auto geometry = block_compressed ? VideoCore::ToBlockCompressedMipGeometry(pitch, height)
                                     : VideoCore::ImageMipGeometry{pitch, height};
    if (pow2) {
        geometry.pitch = std::bit_ceil(geometry.pitch);
        geometry.height = std::bit_ceil(geometry.height);
    }
    const auto [tile_pitch, tile_height, byte_size] =
        VideoCore::ImageSizeMicroTiled(geometry.pitch, geometry.height, 1, bits_per_block, 1);
    const auto padded = block_compressed
                            ? VideoCore::FromBlockCompressedMipGeometry(tile_pitch, tile_height)
                            : VideoCore::ImageMipGeometry{tile_pitch, tile_height};
    ASSERT_LT(pitch, width);
    ASSERT_EQ(padded.pitch, width);
    ASSERT_GT(byte_size, 0U);
    EXPECT_EQ(Vulkan::TextureRejectionReasons(true, true, true), 0U);
}

TEST(TexturePitchAdmission, AcceptsPowerOfTwoPaddedBc1Row) {
    ExpectPaddedImageAdmission(225, 256, 64, true, true);
}

TEST(TexturePitchAdmission, AcceptsPowerOfTwoPaddedBc5Row) {
    ExpectPaddedImageAdmission(193, 256, 128, true, true);
}

TEST(TexturePitchAdmission, AcceptsMicrotilePaddedBc1RowWithoutPowerOfTwoFlag) {
    ExpectPaddedImageAdmission(250, 256, 64, true, false);
}

TEST(TexturePitchAdmission, AcceptsMicrotilePaddedBc5RowWithoutPowerOfTwoFlag) {
    ExpectPaddedImageAdmission(122, 128, 128, true, false);
}

TEST(TexturePitchAdmission, AcceptsPowerOfTwoPaddedUncompressedRow) {
    ExpectPaddedImageAdmission(225, 256, 32, false, true);
}

TEST(TexturePitchAdmission, StillRejectsInvalidAddressAndUnknownFormats) {
    EXPECT_EQ(Vulkan::TextureRejectionReasons(false, true, true), 1U);
    EXPECT_EQ(Vulkan::TextureRejectionReasons(true, false, true), 4U);
    EXPECT_EQ(Vulkan::TextureRejectionReasons(true, true, false), 8U);
    EXPECT_EQ(Vulkan::TextureRejectionReasons(false, false, false), 13U);
    EXPECT_EQ(Vulkan::TextureRejectionReasons(true, true, true), 0U);
}

TEST(TextureFormatAdmission, RejectsUnsupportedSurfaceFormatPairs) {
    using AmdGpu::DataFormat;
    using AmdGpu::NumberFormat;
    EXPECT_FALSE(Vulkan::LiverpoolToVK::TrySurfaceFormat(DataFormat::FormatGB_GR,
                                                          NumberFormat::Sint));
    EXPECT_FALSE(Vulkan::LiverpoolToVK::TrySurfaceFormat(DataFormat::FormatBG_RG,
                                                          NumberFormat::Sint));
    EXPECT_FALSE(Vulkan::LiverpoolToVK::TrySurfaceFormat(DataFormat::FormatFmask8_S2_F1,
                                                          NumberFormat::Uint));
    ASSERT_TRUE(Vulkan::LiverpoolToVK::TrySurfaceFormat(DataFormat::Format8,
                                                         NumberFormat::Unorm));
    EXPECT_EQ(*Vulkan::LiverpoolToVK::TrySurfaceFormat(DataFormat::Format8,
                                                       NumberFormat::Unorm),
              vk::Format::eR8Unorm);
}

TEST(TextureFormatAdmission, RejectsOutOfRangeSurfaceFormatEnumsWithoutAsserting) {
    const auto invalid_data = static_cast<AmdGpu::DataFormat>(63U + 1U);
    const auto invalid_number = static_cast<AmdGpu::NumberFormat>(15U + 1U);
    EXPECT_FALSE(Vulkan::LiverpoolToVK::TrySurfaceFormat(invalid_data,
                                                          AmdGpu::NumberFormat::Unorm));
    EXPECT_FALSE(Vulkan::LiverpoolToVK::TrySurfaceFormat(AmdGpu::DataFormat::Format8,
                                                          invalid_number));
}

TEST(TextureFormatAdmission, UnsupportedPairUsesDedicatedReasonBit) {
    EXPECT_EQ(Vulkan::TextureRejectionReasons(true, true, true, true, false), 0x20U);
    EXPECT_EQ(Vulkan::TextureRejectionReasons(false, false, false, false, false), 0x3dU);
    EXPECT_EQ(Vulkan::TextureRejectionReasons(true, true, true, true, true, false), 0x40U);
    EXPECT_EQ(Vulkan::TextureRejectionReasons(false, false, false, false, false, false), 0x7dU);
}

TEST(TexturePitchAdmission, BindTexturesDoesNotVetoRawPitchBeforeLayout) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto begin = source.find("void Rasterizer::BindTextures(");
    ASSERT_NE(begin, std::string::npos);
    const auto call = source.find("TextureRejectionReasons(", begin);
    ASSERT_NE(call, std::string::npos);
    const auto call_end = source.find(';', call);
    ASSERT_NE(call_end, std::string::npos);
    EXPECT_EQ(source.substr(call, call_end - call).find("tsharp.pitch < tsharp.width"),
              std::string::npos);
}

TEST(TextureFormatAdmission, ValidatesSurfaceFormatPairBeforeFindImage) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto begin = source.find("void Rasterizer::BindTextures(");
    ASSERT_NE(begin, std::string::npos);
    const auto pair = source.find("TrySurfaceFormat(data_fmt, num_fmt)", begin);
    ASSERT_NE(pair, std::string::npos);
    const auto find_image = source.find("image_id = texture_cache.FindImage(desc);", pair);
    ASSERT_NE(find_image, std::string::npos);
    EXPECT_LT(pair, find_image);
}

TEST(TextureFormatAdmission, ValidatesMsaaSampleExponentBeforeFindImage) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto begin = source.find("void Rasterizer::BindTextures(");
    ASSERT_NE(begin, std::string::npos);
    const auto helper = source.find("HasValidMsaaSampleExponent", begin);
    ASSERT_NE(helper, std::string::npos);
    const auto find_image = source.find("image_id = texture_cache.FindImage(desc);", helper);
    ASSERT_NE(find_image, std::string::npos);
    EXPECT_LT(helper, find_image);
    const auto reasons = source.find("TextureRejectionReasons(", helper);
    ASSERT_NE(reasons, std::string::npos);
    EXPECT_LT(reasons, find_image);
}

TEST(SampleCountAdmission, InvalidGuestCountsFailClosedWithoutAsserting) {
    using Vulkan::LiverpoolToVK::NumSamples;

    // A guest image descriptor can carry an arbitrary MSAA last_level.  The
    // Vulkan sample-count domain is discrete and only exposes 1/2/4/8/16;
    // malformed values must not reach an assertion in pipeline/image setup.
    constexpr auto one_sample = vk::SampleCountFlagBits::e1;
    for (const u32 samples : {0U, 3U, 5U, 6U, 7U, 32U, 255U}) {
        EXPECT_EQ(NumSamples(samples, vk::SampleCountFlags{one_sample}), one_sample)
            << "guest sample count=" << samples;
    }
}

TEST(SampleCountAdmission, ValidCountsStillChooseTheLargestSupportedFallback) {
    using Vulkan::LiverpoolToVK::NumSamples;

    constexpr auto one_or_two = vk::SampleCountFlags{vk::SampleCountFlagBits::e1} |
                                 vk::SampleCountFlagBits::e2;
    EXPECT_EQ(NumSamples(1, one_or_two), vk::SampleCountFlagBits::e1);
    EXPECT_EQ(NumSamples(2, one_or_two), vk::SampleCountFlagBits::e2);
    EXPECT_EQ(NumSamples(4, one_or_two), vk::SampleCountFlagBits::e2);
    EXPECT_EQ(NumSamples(16, one_or_two), vk::SampleCountFlagBits::e2);
}

TEST(SampleCountAdmission, InvalidSampleHandlingIsExplicitlyFailClosed) {
    const auto source = ReadSource(SHADPS4_LIVERPOOL_TO_VK_SOURCE_PATH);
    const auto helper = source.find("vk::SampleCountFlagBits RawNumSamples");
    ASSERT_NE(helper, std::string::npos);
    const auto end = source.find("vk::SampleCountFlagBits NumSamples", helper);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(helper, end - helper);
    EXPECT_NE(body.find("Invalid sample count"), std::string::npos);
    EXPECT_EQ(body.find("UNREACHABLE"), std::string::npos);
}

TEST(ImageBufferTransferAdmission, CheckedFootprintRejectsZeroAndOverflow) {
    using VideoCore::CheckedImageBufferTransferSize;

    EXPECT_EQ(CheckedImageBufferTransferSize(0, 1080, 1, 1, 32), std::nullopt);
    EXPECT_EQ(CheckedImageBufferTransferSize(1920, 1080, 1, 1, 0), std::nullopt);
    EXPECT_EQ(CheckedImageBufferTransferSize(1920, 1080, 1, 1, 31), std::nullopt);
    EXPECT_EQ(CheckedImageBufferTransferSize(std::numeric_limits<u64>::max(), 2, 1, 1, 8),
              std::nullopt);
    EXPECT_EQ(CheckedImageBufferTransferSize(std::numeric_limits<u64>::max(), 1, 1, 1, 8),
              std::numeric_limits<u64>::max() / 1);
    EXPECT_EQ(CheckedImageBufferTransferSize(1920, 1080, 1, 1, 32), 1920ULL * 1080ULL * 4ULL);
    EXPECT_TRUE(VideoCore::IsImageBufferTransferRangeWithinBuffer(0, 4096, 4096));
    EXPECT_TRUE(VideoCore::IsImageBufferTransferRangeWithinBuffer(1024, 1024, 2048));
    EXPECT_FALSE(VideoCore::IsImageBufferTransferRangeWithinBuffer(1025, 1024, 2048));
    EXPECT_FALSE(VideoCore::IsImageBufferTransferRangeWithinBuffer(
        std::numeric_limits<u64>::max(), 1, std::numeric_limits<u64>::max()));
}

TEST(ImageBufferTransferAdmission, DepthStencilFootprintIsCheckedBeforePublication) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_INFO_PATH);
    const auto constructor = source.find("ImageInfo::ImageInfo(const AmdGpu::DepthBuffer& buffer");
    ASSERT_NE(constructor, std::string::npos);
    const auto checked = source.find("CheckedImageBufferTransferSize(", constructor);
    const auto assignment = source.find("stencil_size =", constructor);
    ASSERT_NE(checked, std::string::npos);
    ASSERT_NE(assignment, std::string::npos);
    EXPECT_LT(checked, assignment);
    EXPECT_NE(source.substr(checked).find("IsGuestImageFootprintRepresentable"), std::string::npos);
}

TEST(ImageBufferTransferAdmission, ReadbackSkipsUnrepresentableFootprintBeforeMapping) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_TEXTURE_CACHE_PATH);
    const auto download = source.find("void TextureCache::DownloadImageMemory(");
    ASSERT_NE(download, std::string::npos);
    const auto checked = source.find("CheckedImageBufferTransferSize(", download);
    const auto capacity = source.find("download_buffer.SizeBytes()", download);
    const auto map = source.find("download_buffer.Map(", download);
    const auto assertion = source.find("ASSERT(download_size", download);
    ASSERT_NE(checked, std::string::npos);
    ASSERT_NE(capacity, std::string::npos);
    ASSERT_NE(map, std::string::npos);
    EXPECT_LT(checked, map);
    EXPECT_LT(capacity, map);
    EXPECT_EQ(assertion, std::string::npos);
}

TEST(ImageBufferTransferAdmission, BufferBackedDepthCopyChecksEveryRegionBeforeRecording) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_PATH);
    const auto copy = source.find("void Image::CopyImageWithBuffer(");
    ASSERT_NE(copy, std::string::npos);
    const auto range_check = source.find("IsImageBufferTransferRangeWithinBuffer", copy);
    const auto image_to_buffer = source.find("copyImageToBuffer", copy);
    const auto buffer_to_image = source.find("copyBufferToImage", copy);
    ASSERT_NE(range_check, std::string::npos);
    ASSERT_NE(image_to_buffer, std::string::npos);
    ASSERT_NE(buffer_to_image, std::string::npos);
    EXPECT_LT(range_check, image_to_buffer);
    EXPECT_LT(range_check, buffer_to_image);
}

TEST(TextureRejectionDiagnosticWiring, ReadsOnlyOwnedSharpInsideExistingRejection) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto begin = source.find("void Rasterizer::BindTextures(");
    ASSERT_NE(begin, std::string::npos);
    const auto reasons = source.find("TextureRejectionReasons(", begin);
    ASSERT_NE(reasons, std::string::npos);
    const auto observation = source.find("const auto build_texture_observation", reasons);
    ASSERT_NE(observation, std::string::npos);
    const auto copy = source.find("std::memcpy(observation.words.data(), &tsharp", observation);
    ASSERT_NE(copy, std::string::npos);
    const auto null_binding =
        source.find("append_invalid_image_binding(image_desc.is_written);", copy);
    ASSERT_NE(null_binding, std::string::npos);
    const auto diagnostic = source.substr(observation, null_binding - observation);
    EXPECT_NE(diagnostic.find("rejection_diagnostics.Observe(build_texture_observation)"),
              std::string::npos);
    EXPECT_EQ(diagnostic.find("ReadMemory"), std::string::npos);
    EXPECT_EQ(diagnostic.find("ReadUdSharp"), std::string::npos);
    for (const auto field : {"stage.pgm_hash", "image_desc.sharp_idx", "array_index = i",
                             "descriptor_count = num_bindings", "tsharp.pow2pad",
                             "tsharp.tiling_index", "tsharp.base_level", "tsharp.last_level"}) {
        EXPECT_NE(diagnostic.find(field), std::string::npos) << field;
    }
}

TEST(TextureDiagnosticContext, DisabledDoesNotEvaluateBuilder) {
    unsigned builds = 0;
    Vulkan::ScopedTextureDiagnosticContext context{false, [&] {
                                                       ++builds;
                                                       return Example();
                                                   }};
    EXPECT_EQ(Vulkan::ScopedTextureDiagnosticContext::Current(), nullptr);
    EXPECT_EQ(builds, 0U);
}

TEST(TextureDiagnosticContext, OwnedContextIsAvailableUntilScopeExit) {
    auto source = Example();
    {
        Vulkan::ScopedTextureDiagnosticContext context{true, [&] { return source; }};
        ASSERT_NE(Vulkan::ScopedTextureDiagnosticContext::Current(), nullptr);
        source.words.fill(0);
        EXPECT_EQ(*Vulkan::ScopedTextureDiagnosticContext::Current(), Example());
    }
    EXPECT_EQ(Vulkan::ScopedTextureDiagnosticContext::Current(), nullptr);
}

TEST(TextureDiagnosticContext, NestedContextRestoresOuterOnUnwind) {
    Vulkan::ScopedTextureDiagnosticContext outer{true, Example};
    ASSERT_NE(Vulkan::ScopedTextureDiagnosticContext::Current(), nullptr);
    const auto* outer_pointer = Vulkan::ScopedTextureDiagnosticContext::Current();
    try {
        auto inner_value = Example();
        inner_value.array_index = 12;
        Vulkan::ScopedTextureDiagnosticContext inner{true, [&] { return inner_value; }};
        ASSERT_NE(Vulkan::ScopedTextureDiagnosticContext::Current(), nullptr);
        EXPECT_EQ(Vulkan::ScopedTextureDiagnosticContext::Current()->array_index, 12U);
        throw 1;
    } catch (int) {
    }
    EXPECT_EQ(Vulkan::ScopedTextureDiagnosticContext::Current(), outer_pointer);
    EXPECT_EQ(*outer_pointer, Example());
}

TEST(TextureRejectionDiagnosticWiring, AdmittedDescriptorContextSurroundsSynchronousFindImage) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH);
    const auto binding = source.find("void Rasterizer::BindTextures(");
    ASSERT_NE(binding, std::string::npos);
    const auto context = source.find("ScopedTextureDiagnosticContext diagnostic_context", binding);
    ASSERT_NE(context, std::string::npos);
    const auto find = source.find("image_id = texture_cache.FindImage(desc);", context);
    ASSERT_NE(find, std::string::npos);
    const auto image_source = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_PATH);
    const auto fail =
        image_source.find("result != VK_SUCCESS && TextureRejectionDiagnosticsEnabled()");
    ASSERT_NE(fail, std::string::npos);
    const auto read_context = image_source.find("ScopedTextureDiagnosticContext::Current()", fail);
    ASSERT_NE(read_context, std::string::npos);
    EXPECT_LT(read_context, image_source.find("ASSERT_MSG(result == VK_SUCCESS", fail));
}

TEST(TextureRejectionDiagnosticWiring, AllocationFailureLogsActualCreateInfoBeforeAssertion) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_PATH);
    const auto create = source.find("vk::Result UniqueImage::TryCreate(");
    ASSERT_NE(create, std::string::npos);
    const auto fail =
        source.find("result != VK_SUCCESS && TextureRejectionDiagnosticsEnabled()", create);
    ASSERT_NE(fail, std::string::npos);
    const auto assertion = source.find("return vk::Result{result};", fail);
    ASSERT_NE(assertion, std::string::npos);
    const auto diagnostic = source.substr(fail, assertion - fail);
    for (const auto field :
         {"image_ci.extent.width", "image_ci.extent.height", "image_ci.extent.depth",
          "image_ci.mipLevels", "image_ci.arrayLayers", "image_ci.format", "image_ci.usage",
          "image_ci.samples", "image_ci.flags"}) {
        EXPECT_NE(diagnostic.find(field), std::string::npos) << field;
    }
    EXPECT_EQ(diagnostic.find("getImageMemoryRequirements"), std::string::npos);
}

TEST(TextureImageAdmission, AllocationFailureDoesNotPublishTextureCacheBacking) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_PATH);
    const auto constructor = source.find("Image::Image(const Vulkan::Instance& instance_");
    ASSERT_NE(constructor, std::string::npos);
    const auto try_create = source.find("backing->image.TryCreate(image_ci, instance)", constructor);
    ASSERT_NE(try_create, std::string::npos);
    const auto rollback = source.find("backing_images.pop_back()", try_create);
    const auto clear_backing = source.find("backing = nullptr", try_create);
    const auto return_path = source.find("return;", try_create);
    ASSERT_NE(rollback, std::string::npos);
    ASSERT_NE(clear_backing, std::string::npos);
    ASSERT_NE(return_path, std::string::npos);
    EXPECT_LT(rollback, return_path);
    EXPECT_LT(clear_backing, return_path);
}

TEST(TextureImageAdmission, DepthOverlapRebuildsFootprintAfterResourceUnion) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_TEXTURE_CACHE_PATH);
    const auto resolve = source.find("ImageId TextureCache::ResolveDepthOverlap(");
    ASSERT_NE(resolve, std::string::npos);
    const auto merge = source.find("MergeSubresourceExtents(", resolve);
    const auto recompute = source.find("new_info.UpdateSize()", merge);
    const auto insert = source.find("InsertImage(new_info", recompute);
    ASSERT_NE(merge, std::string::npos);
    ASSERT_NE(recompute, std::string::npos);
    ASSERT_NE(insert, std::string::npos);
    EXPECT_LT(merge, recompute);
    EXPECT_LT(recompute, insert);
}

TEST(TextureImageAdmission, SampledMsaaBackingRevalidatesUsageBeforeCreation) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_PATH);
    const auto set_backing = source.find("void Image::SetBackingSamples(");
    ASSERT_NE(set_backing, std::string::npos);
    const auto policy = source.find("SelectImageUsageForSamples(", set_backing);
    const auto assign_usage = source.find("new_image_ci.usage = selected_usage->usage;", policy);
    const auto try_create = source.find("TryCreate(new_image_ci, instance)", assign_usage);
    ASSERT_NE(policy, std::string::npos);
    ASSERT_NE(assign_usage, std::string::npos);
    ASSERT_NE(try_create, std::string::npos);
    EXPECT_LT(policy, assign_usage);
    EXPECT_LT(assign_usage, try_create);
}

TEST(TextureImageAdmission, ArrayedMsaaUsesTheTwoDimensionalHostImageType) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_PATH);
    const auto conversion = source.find("static vk::ImageType ConvertImageType(");
    ASSERT_NE(conversion, std::string::npos);
    const auto array_msaa = source.find("AmdGpu::ImageType::Color2DMsaaArray", conversion);
    const auto unreachable = source.find("UNREACHABLE()", conversion);
    ASSERT_NE(array_msaa, std::string::npos);
    ASSERT_NE(unreachable, std::string::npos);
    EXPECT_LT(array_msaa, unreachable);
}

TEST(TextureImageAdmission, ArrayedMsaaViewUsesTwoDimensionalArrayView) {
    const auto source = ReadSource(SHADPS4_TEXTURE_REJECTION_IMAGE_VIEW_PATH);
    const auto conversion = source.find("vk::ImageViewType ConvertImageViewType(");
    ASSERT_NE(conversion, std::string::npos);
    const auto array_msaa =
        source.find("case AmdGpu::ImageType::Color2DMsaaArray:", conversion);
    const auto array_return =
        source.find("return vk::ImageViewType::e2DArray;", array_msaa);
    const auto unreachable = source.find("UNREACHABLE()", conversion);
    ASSERT_NE(array_msaa, std::string::npos);
    ASSERT_NE(array_return, std::string::npos);
    ASSERT_NE(unreachable, std::string::npos);
    EXPECT_LT(array_msaa, array_return);
    EXPECT_LT(array_return, unreachable);
}

} // namespace
