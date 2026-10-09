// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/texture_rejection_diagnostics.h"

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
            .reasons = 2,
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

TEST(TextureRejectionDiagnostics, ReasonMaskPreservesAllExistingGuardOutcomes) {
    for (unsigned bits = 0; bits < 16; ++bits) {
        const bool address_40bit = !(bits & 1);
        const bool short_pitch = bits & 2;
        const bool valid_data = !(bits & 4);
        const bool valid_number = !(bits & 8);
        const auto reasons =
            Vulkan::TextureRejectionReasons(address_40bit, short_pitch, valid_data, valid_number);
        EXPECT_EQ(reasons, bits);
        EXPECT_EQ(reasons != 0, !address_40bit || short_pitch || !valid_data || !valid_number);
    }
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
