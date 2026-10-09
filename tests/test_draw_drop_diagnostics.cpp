// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "video_core/renderer_vulkan/draw_drop_diagnostics.h"

namespace {

using Vulkan::DescriptorImageCommandContext;
using Vulkan::DescriptorImageCommandKind;
using Vulkan::DrawDropDiagnostics;
using Vulkan::DrawDropDiagnosticsPolicy;
using Vulkan::DrawDropReason;

constexpr DescriptorImageCommandContext DrawContext() {
    return {
        .kind = DescriptorImageCommandKind::DirectDraw,
        .vertex_hash = 0x1111,
        .fragment_hash = 0x2222,
    };
}

#ifdef SHADPS4_DRAW_DROP_RASTERIZER_PATH
std::string ReadSource() {
    std::ifstream stream{SHADPS4_DRAW_DROP_RASTERIZER_PATH};
    if (!stream) {
        stream.clear();
        stream.open("src/video_core/renderer_vulkan/vk_rasterizer.cpp");
    }
    if (!stream) {
        stream.clear();
        stream.open("../src/video_core/renderer_vulkan/vk_rasterizer.cpp");
    }
    return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}
#endif

TEST(DrawDropDiagnostics, RequiresStrictExplicitBoundedOptIn) {
    EXPECT_FALSE(DrawDropDiagnosticsPolicy::Parse(std::nullopt, "0", "1").enabled);
    EXPECT_FALSE(DrawDropDiagnosticsPolicy::Parse("0", "0", "1").enabled);
    EXPECT_FALSE(DrawDropDiagnosticsPolicy::Parse("1", std::nullopt, "1").valid);
    EXPECT_FALSE(DrawDropDiagnosticsPolicy::Parse("1", "0", std::nullopt).valid);
    EXPECT_FALSE(DrawDropDiagnosticsPolicy::Parse("yes", "0", "1").valid);
    EXPECT_FALSE(DrawDropDiagnosticsPolicy::Parse("1", "600001", "1").valid);
    EXPECT_FALSE(DrawDropDiagnosticsPolicy::Parse("1", "0", "5001").valid);

    const auto policy = DrawDropDiagnosticsPolicy::Parse("1", "12", "345");
    ASSERT_TRUE(policy.valid);
    EXPECT_TRUE(policy.enabled);
    EXPECT_EQ(policy.start_ms, 12U);
    EXPECT_EQ(policy.duration_ms, 345U);
}

TEST(DrawDropDiagnostics, CountsNullPipelineAndResourceDropsByCommand) {
    DrawDropDiagnostics diagnostics{{true, true, 0, 5000}};

    EXPECT_TRUE(diagnostics.Observe(0, DrawDropReason::NullPipeline, DrawContext()));
    EXPECT_TRUE(diagnostics.Observe(
        1, DrawDropReason::ResourceBinding,
        DescriptorImageCommandContext{.kind = DescriptorImageCommandKind::IndexedDraw,
                                      .vertex_hash = 0x3333,
                                      .fragment_hash = 0x4444}));
    EXPECT_TRUE(diagnostics.Observe(2, DrawDropReason::ResourceBinding, DrawContext()));

    const auto summary = diagnostics.Close(3);
    EXPECT_EQ(summary.total, 3U);
    EXPECT_EQ(summary.Count(DrawDropReason::NullPipeline, DescriptorImageCommandKind::DirectDraw),
              1U);
    EXPECT_EQ(
        summary.Count(DrawDropReason::ResourceBinding, DescriptorImageCommandKind::IndexedDraw),
        1U);
    EXPECT_EQ(summary.Count(DrawDropReason::ResourceBinding,
                            DescriptorImageCommandKind::DirectDraw),
              1U);
    EXPECT_FALSE(summary.saturated);
}

TEST(DrawDropDiagnostics, WindowIsStartInclusiveAndEndExclusive) {
    DrawDropDiagnostics diagnostics{{true, true, 100, 10}};

    EXPECT_FALSE(diagnostics.Observe(99, DrawDropReason::NullPipeline, DrawContext()));
    EXPECT_TRUE(diagnostics.Observe(100, DrawDropReason::NullPipeline, DrawContext()));
    EXPECT_TRUE(diagnostics.Observe(109, DrawDropReason::NullPipeline, DrawContext()));
    EXPECT_FALSE(diagnostics.Observe(110, DrawDropReason::NullPipeline, DrawContext()));
    EXPECT_TRUE(diagnostics.Armed());

    const auto summary = diagnostics.Close(111);
    EXPECT_EQ(summary.total, 2U);
    EXPECT_EQ(summary.Count(DrawDropReason::NullPipeline, DescriptorImageCommandKind::DirectDraw),
              2U);
}

TEST(DrawDropDiagnostics, FixedCapacitySaturatesWithoutGrowing) {
    DrawDropDiagnostics diagnostics{{true, true, 0, 5000}};
    for (std::size_t index = 0; index < DrawDropDiagnostics::MaxEvents; ++index) {
        EXPECT_TRUE(diagnostics.Observe(
            index, DrawDropReason::ResourceBinding,
            DescriptorImageCommandContext{.kind = DescriptorImageCommandKind::IndirectDraw}));
    }
    EXPECT_FALSE(diagnostics.Observe(
        DrawDropDiagnostics::MaxEvents, DrawDropReason::NullPipeline,
        DescriptorImageCommandContext{.kind = DescriptorImageCommandKind::IndirectDraw}));

    const auto summary = diagnostics.Close(DrawDropDiagnostics::MaxEvents + 1);
    EXPECT_EQ(summary.total, DrawDropDiagnostics::MaxEvents);
    EXPECT_TRUE(summary.saturated);
    EXPECT_EQ(summary.Count(DrawDropReason::ResourceBinding,
                            DescriptorImageCommandKind::IndirectDraw),
              DrawDropDiagnostics::MaxEvents);
    EXPECT_EQ(summary.Count(DrawDropReason::NullPipeline,
                            DescriptorImageCommandKind::IndirectDraw),
              0U);
}

TEST(DrawDropDiagnostics, DisabledDoesNotRetainOrCount) {
    DrawDropDiagnostics diagnostics{{}};
    EXPECT_FALSE(diagnostics.Observe(0, DrawDropReason::NullPipeline, DrawContext()));
    const auto summary = diagnostics.Close(1);
    EXPECT_EQ(summary.total, 0U);
    EXPECT_EQ(summary.Count(DrawDropReason::NullPipeline, DescriptorImageCommandKind::DirectDraw),
              0U);
}

#ifdef SHADPS4_DRAW_DROP_RASTERIZER_PATH
TEST(DrawDropDiagnostics, ProductionWiringIsObservationOnlyAndCoversSilentDropPaths) {
    const auto source = ReadSource();
    for (const auto variable : {"SHADPS4_DRAW_DROP_DIAGNOSTICS",
                                "SHADPS4_DRAW_DROP_DIAGNOSTICS_START_MS",
                                "SHADPS4_DRAW_DROP_DIAGNOSTICS_DURATION_MS"}) {
        EXPECT_NE(source.find(variable), std::string::npos) << variable;
    }
    for (const auto marker : {"void Rasterizer::ObserveDrawDrop(",
                              "DrawDropReason::NullPipeline",
                              "DrawDropReason::ResourceBinding",
                              "DrawDropDiagnosticsV1", "behavior=observation_only"}) {
        EXPECT_NE(source.find(marker), std::string::npos) << marker;
    }
}
#endif

} // namespace
