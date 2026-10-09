// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/texture_rejection_diagnostics.h"
#include "video_core/texture_cache/image_layout_policy.h"
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
    const auto create = source.find("void UniqueImage::Create(");
    ASSERT_NE(create, std::string::npos);
    const auto fail =
        source.find("result != VK_SUCCESS && TextureRejectionDiagnosticsEnabled()", create);
    ASSERT_NE(fail, std::string::npos);
    const auto assertion = source.find("ASSERT_MSG(result == VK_SUCCESS", fail);
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

} // namespace
