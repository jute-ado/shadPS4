// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/sampler_rejection_diagnostics.h"

namespace {

using Vulkan::SamplerRejectionDiagnostics;
using Vulkan::SamplerRejectionObservation;

SamplerRejectionObservation Example() {
    return {.shader_hash = 0x123456789abcdef0ULL,
            .stage = 1,
            .logical_stage = 3,
            .binding = 4,
            .sharp_index = 72,
            .reasons = 1,
            .is_inline = false,
            .disable_aniso = true,
            .host_anisotropy_supported = true,
            .associated_image = 2,
            .associated_sharp_index = 80,
            .associated_descriptor_dwords = 8,
            .words = {0x10000e00, 0x10203040, 0x00a00000, 0x50607080},
            .effective_words = {0x10000e00, 0x10203040, 0x00a00000, 0x50607080},
            .associated_words = {1, 2, 3, 4, 5, 6, 7, 8}};
}

TEST(SamplerRejectionDiagnostics, DisabledDoesNotBuildOrRetainTuple) {
    SamplerRejectionDiagnostics diagnostics{false};
    unsigned builds = 0;
    EXPECT_FALSE(diagnostics.Observe([&] {
        ++builds;
        return Example();
    }));
    EXPECT_EQ(builds, 0U);
    EXPECT_EQ(diagnostics.RecordedCount(), 0U);
}

TEST(SamplerRejectionDiagnostics, PreservesRawEffectiveAndAssociatedWords) {
    SamplerRejectionDiagnostics diagnostics{true};
    const auto actual = diagnostics.Observe(Example);
    ASSERT_TRUE(actual);
    EXPECT_EQ(*actual, Example());
    EXPECT_EQ(diagnostics.RecordedCount(), 1U);
}

TEST(SamplerRejectionDiagnostics, DuplicateDoesNotConsumeBudget) {
    SamplerRejectionDiagnostics diagnostics{true};
    ASSERT_TRUE(diagnostics.Observe(Example));
    for (unsigned repeat = 0; repeat < 100; ++repeat) {
        EXPECT_FALSE(diagnostics.Observe(Example));
    }
    EXPECT_EQ(diagnostics.RecordedCount(), 1U);
}

TEST(SamplerRejectionDiagnostics, EveryRawWordParticipatesInDeduplication) {
    SamplerRejectionDiagnostics diagnostics{true};
    ASSERT_TRUE(diagnostics.Observe(Example));
    for (unsigned word = 0; word < 16; ++word) {
        auto value = Example();
        if (word < 4) {
            value.words[word] ^= 1U;
        } else if (word < 8) {
            value.effective_words[word - 4] ^= 1U;
        } else {
            value.associated_words[word - 8] ^= 1U;
        }
        EXPECT_TRUE(diagnostics.Observe([&] { return value; }));
    }
    EXPECT_EQ(diagnostics.RecordedCount(), 17U);
}

TEST(SamplerRejectionDiagnostics, EveryContextFieldParticipatesInDeduplication) {
    SamplerRejectionDiagnostics diagnostics{true};
    ASSERT_TRUE(diagnostics.Observe(Example));
    for (unsigned field = 0; field < 12; ++field) {
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
            ++value.binding;
            break;
        case 4:
            ++value.sharp_index;
            break;
        case 5:
            ++value.reasons;
            break;
        case 6:
            value.is_inline = !value.is_inline;
            break;
        case 7:
            value.disable_aniso = !value.disable_aniso;
            break;
        case 8:
            value.host_anisotropy_supported = !value.host_anisotropy_supported;
            break;
        case 9:
            ++value.associated_image;
            break;
        case 10:
            ++value.associated_sharp_index;
            break;
        case 11:
            ++value.associated_descriptor_dwords;
            break;
        }
        EXPECT_TRUE(diagnostics.Observe([&] { return value; }));
    }
    EXPECT_EQ(diagnostics.RecordedCount(), 13U);
}

TEST(SamplerRejectionDiagnostics, OwnsTupleAcrossSourceMutation) {
    SamplerRejectionDiagnostics diagnostics{true};
    auto source = Example();
    const auto first = diagnostics.Observe([&] { return source; });
    ASSERT_TRUE(first);
    source.words.fill(0);
    source.associated_words.fill(0);
    EXPECT_EQ(*first, Example());
    EXPECT_FALSE(diagnostics.Observe(Example));
    EXPECT_TRUE(diagnostics.Observe([&] { return source; }));
}

TEST(SamplerRejectionDiagnostics, ExhaustedBudgetDoesNotBuildTuple) {
    EXPECT_EQ(SamplerRejectionDiagnostics::MaxObservations, 32U);
    SamplerRejectionDiagnostics diagnostics{true};
    for (unsigned index = 0; index < SamplerRejectionDiagnostics::MaxObservations; ++index) {
        auto value = Example();
        value.binding = index;
        ASSERT_TRUE(diagnostics.Observe([&] { return value; }));
    }
    unsigned builds = 0;
    EXPECT_FALSE(diagnostics.Observe([&] {
        ++builds;
        return Example();
    }));
    EXPECT_EQ(builds, 0U);
    EXPECT_EQ(diagnostics.RecordedCount(), SamplerRejectionDiagnostics::MaxObservations);
}

TEST(SamplerRejectionDiagnostics, ReportsOnlyUnsupportedDecoderFieldsWithoutMutatingWords) {
    for (unsigned aniso = 0; aniso < 8; ++aniso) {
        for (unsigned filter = 0; filter < 4; ++filter) {
            for (unsigned mip = 0; mip < 4; ++mip) {
                const std::uint64_t raw0 =
                    (std::uint64_t{aniso} << 9) | (std::uint64_t{filter} << 29);
                const std::uint64_t raw1 = std::uint64_t{mip} << 26;
                const unsigned expected =
                    (aniso > 4 ? 1U : 0U) | (filter > 2 ? 2U : 0U) | (mip > 2 ? 4U : 0U);
                EXPECT_EQ(Vulkan::SamplerRejectionReasons(raw0, raw1), expected);
                // Unrelated sampler fields, including XY filters, do not change the report.
                EXPECT_EQ(Vulkan::SamplerRejectionReasons(raw0 | 0x1ffff1ffULL,
                                                          raw1 | 0xfffffffff3ffffffULL),
                          expected);
            }
        }
    }
}

TEST(SamplerRejectionDiagnosticWiring, ObservesOwnedDataOnlyBeforeUnchangedSamplerConversion) {
    std::ifstream input{SHADPS4_SAMPLER_REJECTION_RASTERIZER_PATH, std::ios::binary};
    ASSERT_TRUE(input.is_open());
    std::string source{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    std::erase(source, '\r');
    const auto begin = source.find("for (const auto& sampler : stage.samplers)");
    ASSERT_NE(begin, std::string::npos);
    const auto conversion =
        source.find("texture_cache.GetSampler(ssharp, liverpool->regs.ta_bc_base)", begin);
    ASSERT_NE(conversion, std::string::npos);
    const auto diagnostic = source.find("if (TextureRejectionDiagnosticsEnabled())", begin);
    ASSERT_LT(diagnostic, conversion);
    const auto scope = source.substr(diagnostic, conversion - diagnostic);
    for (const auto field :
         {"SamplerRejectionReasons(ssharp.raw0, ssharp.raw1)", "sampler_diagnostics.Observe(",
          "stage.pgm_hash", "binding.unified", "sampler.sharp_idx", "sampler.is_inline_sampler",
          "sampler.disable_aniso", "sampler.associated_image", "stage.flattened_ud_buf.data()",
          "observation.words.data()", "observation.effective_words.data()",
          "observation.associated_words.data()"}) {
        EXPECT_NE(scope.find(field), std::string::npos) << field;
    }
    for (const auto forbidden : {"ReadGuestMemory", "ReadMemory", "ReadUdReg", "CopySparseMemory",
                                 "return;", "continue;", "ssharp.max_aniso.Assign"}) {
        EXPECT_EQ(scope.find(forbidden), std::string::npos) << forbidden;
    }
    EXPECT_LT(source.find("ssharp.max_aniso.Assign", begin), diagnostic);
}

} // namespace
