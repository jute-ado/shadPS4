// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>

#include "video_core/amdgpu/input_assembly_device_readback_plan.h"

namespace {

using AmdGpu::BoundInputAssemblySource;
using AmdGpu::InputAssemblyAuthority;
using AmdGpu::InputAssemblyBufferToken;
using AmdGpu::InputAssemblyCopyAccess;
using AmdGpu::InputAssemblyDeviceReadbackPlanner;
using AmdGpu::InputAssemblyHostUsage;
using AmdGpu::InputAssemblyReadbackCompletion;
using AmdGpu::InputAssemblySemanticOrdinal;
using AmdGpu::InputAssemblySourceKind;

constexpr InputAssemblyBufferToken Buffer(u32 slot, u32 generation) {
    return {.slot = slot, .generation = generation};
}

constexpr InputAssemblySemanticOrdinal Semantic(u32 draw, InputAssemblySourceKind kind,
                                                u32 binding) {
    return {.draw = draw, .kind = kind, .binding = binding};
}

constexpr BoundInputAssemblySource Source(InputAssemblyHostUsage usage, u64 offset = 64,
                                          u64 size = 4096) {
    return {
        .token = Buffer(7, 3),
        .usage = usage,
        .host_offset = offset,
        .host_size = size,
        .write_serial = 9,
        .authority = InputAssemblyAuthority::GpuAuthoritative,
    };
}

TEST(InputAssemblyDeviceReadbackPlan, TranslatesMergedVertexBindingToActualHostBytes) {
    const auto normalized = AmdGpu::NormalizeVertexInputRange(
        Source(InputAssemblyHostUsage::DeviceLocal, 256, 2048), /*merged_guest_base=*/1000,
        /*binding_guest_base=*/1128, /*binding_size=*/96,
        Semantic(17, InputAssemblySourceKind::Vertex, 3));

    ASSERT_TRUE(normalized.has_value());
    EXPECT_EQ(normalized->source_offset, 384);
    EXPECT_EQ(normalized->size, 96);
    EXPECT_EQ(normalized->source_size, 2048);
    EXPECT_EQ(normalized->semantic, Semantic(17, InputAssemblySourceKind::Vertex, 3));
}

TEST(InputAssemblyDeviceReadbackPlan, ChecksIndexOffsetStrideAndExactBoundRange) {
    const auto normalized = AmdGpu::NormalizeIndexInputRange(
        Source(InputAssemblyHostUsage::Stream, 128, 1024), /*index_offset=*/7,
        /*index_stride=*/2, /*index_count=*/30, Semantic(22, InputAssemblySourceKind::Index, 0));
    ASSERT_TRUE(normalized.has_value());
    EXPECT_EQ(normalized->source_offset, 142);
    EXPECT_EQ(normalized->size, 60);

    EXPECT_FALSE(AmdGpu::NormalizeIndexInputRange(Source(InputAssemblyHostUsage::Stream),
                                                  std::numeric_limits<u64>::max(), 4, 1,
                                                  Semantic(1, InputAssemblySourceKind::Index, 0)));
    EXPECT_FALSE(AmdGpu::NormalizeIndexInputRange(Source(InputAssemblyHostUsage::Stream), 0,
                                                  std::numeric_limits<u64>::max(), 2,
                                                  Semantic(1, InputAssemblySourceKind::Index, 0)));
}

TEST(InputAssemblyDeviceReadbackPlan, AdmitsStreamAndDeviceLocalSources) {
    for (const auto usage : {InputAssemblyHostUsage::Stream, InputAssemblyHostUsage::DeviceLocal}) {
        const auto normalized = AmdGpu::NormalizeVertexInputRange(
            Source(usage), 200, 200, 64, Semantic(1, InputAssemblySourceKind::Vertex, 0));
        ASSERT_TRUE(normalized.has_value());
        EXPECT_EQ(normalized->usage, usage);
    }
}

TEST(InputAssemblyDeviceReadbackPlan, DeduplicatesSamplesWhileRetainingBothSemantics) {
    InputAssemblyDeviceReadbackPlanner planner;
    planner.BeginFrame(60);
    const auto vertex =
        AmdGpu::NormalizeVertexInputRange(Source(InputAssemblyHostUsage::DeviceLocal, 100, 512), 0,
                                          0, 64, Semantic(5, InputAssemblySourceKind::Vertex, 2));
    const auto index =
        AmdGpu::NormalizeVertexInputRange(Source(InputAssemblyHostUsage::DeviceLocal, 100, 512), 0,
                                          32, 64, Semantic(5, InputAssemblySourceKind::Index, 0));
    ASSERT_TRUE(vertex && index);

    const auto vertex_decision = planner.Plan(*vertex);
    const auto index_decision = planner.Plan(*index);
    ASSERT_TRUE(vertex_decision.accepted);
    ASSERT_TRUE(index_decision.accepted);
    EXPECT_EQ(vertex_decision.reference_count, 3);
    EXPECT_EQ(index_decision.reference_count, 3);
    EXPECT_EQ(vertex_decision.new_copy_count, 3);
    EXPECT_EQ(index_decision.new_copy_count, 2);
    EXPECT_LT(planner.EndFrame().sample_count,
              vertex_decision.reference_count + index_decision.reference_count);
    EXPECT_EQ(planner.EndFrame().semantic_count, 2);
}

TEST(InputAssemblyDeviceReadbackPlan, AlignsPhysicalCopiesButRetainsExactSemanticSlice) {
    InputAssemblyDeviceReadbackPlanner planner;
    planner.BeginFrame(62);
    const auto normalized =
        AmdGpu::NormalizeVertexInputRange(Source(InputAssemblyHostUsage::Stream, 3, 64), 0, 0, 16,
                                          Semantic(1, InputAssemblySourceKind::Vertex, 0));
    ASSERT_TRUE(normalized);

    const auto decision = planner.Plan(*normalized);
    ASSERT_TRUE(decision.accepted);
    ASSERT_EQ(decision.reference_count, 1);
    ASSERT_EQ(decision.new_copy_count, 1);
    const auto& plan = planner.EndFrame();
    ASSERT_EQ(plan.sample_count, 1);
    EXPECT_EQ(plan.samples[0].source_offset, 0);
    EXPECT_EQ(plan.samples[0].destination_offset, 0);
    EXPECT_EQ(plan.samples[0].size, 20);
    EXPECT_EQ(plan.samples[0].source_offset % 4, 0);
    EXPECT_EQ(plan.samples[0].destination_offset % 4, 0);
    EXPECT_EQ(plan.samples[0].size % 4, 0);
    EXPECT_EQ(decision.references[0].destination_offset, 3);
    EXPECT_EQ(decision.references[0].size, 16);
}

TEST(InputAssemblyDeviceReadbackPlan, RejectsConflictingMetadataForOneSourceGeneration) {
    InputAssemblyDeviceReadbackPlanner planner;
    planner.BeginFrame(63);
    const auto first =
        AmdGpu::NormalizeVertexInputRange(Source(InputAssemblyHostUsage::Stream), 0, 0, 16,
                                          Semantic(1, InputAssemblySourceKind::Vertex, 0));
    auto conflicting_source = Source(InputAssemblyHostUsage::DeviceLocal);
    conflicting_source.authority = InputAssemblyAuthority::CpuAuthoritative;
    const auto conflicting = AmdGpu::NormalizeVertexInputRange(
        conflicting_source, 0, 0, 16, Semantic(2, InputAssemblySourceKind::Vertex, 0));
    ASSERT_TRUE(first && conflicting);
    EXPECT_TRUE(planner.Plan(*first).accepted);
    EXPECT_FALSE(planner.Plan(*conflicting).accepted);
    const auto& plan = planner.EndFrame();
    EXPECT_FALSE(plan.complete);
    EXPECT_EQ(plan.sample_count, 1);
    EXPECT_EQ(plan.semantic_count, 1);
    EXPECT_EQ(plan.loss.source_conflict, 1);
}

TEST(InputAssemblyDeviceReadbackPlan, RejectsInvalidZeroAndOutOfBoundsSources) {
    auto invalid = Source(InputAssemblyHostUsage::DeviceLocal);
    invalid.token = {};
    EXPECT_FALSE(AmdGpu::NormalizeVertexInputRange(
        invalid, 0, 0, 32, Semantic(1, InputAssemblySourceKind::Vertex, 0)));
    EXPECT_FALSE(
        AmdGpu::NormalizeVertexInputRange(Source(InputAssemblyHostUsage::DeviceLocal), 0, 0, 0,
                                          Semantic(1, InputAssemblySourceKind::Vertex, 0)));
    EXPECT_FALSE(AmdGpu::NormalizeVertexInputRange(
        Source(InputAssemblyHostUsage::DeviceLocal, 4090, 4096), 0, 0, 16,
        Semantic(1, InputAssemblySourceKind::Vertex, 0)));
}

TEST(InputAssemblyDeviceReadbackPlan, EnforcesFixedSampleAndByteCapsTransactionally) {
    InputAssemblyDeviceReadbackPlanner planner;
    planner.BeginFrame(61);
    for (u32 i = 0; i < InputAssemblyDeviceReadbackPlanner::MaxSamplesPerFrame; ++i) {
        auto source = Source(InputAssemblyHostUsage::Stream, u64{i} * 64, u64{i + 1} * 64);
        source.token = Buffer(i + 1, 1);
        const auto normalized = AmdGpu::NormalizeVertexInputRange(
            source, 0, 0, 16, Semantic(i, InputAssemblySourceKind::Vertex, 0));
        ASSERT_TRUE(normalized);
        EXPECT_TRUE(planner.Plan(*normalized).accepted) << i;
    }
    auto overflow_source = Source(InputAssemblyHostUsage::Stream);
    overflow_source.token = Buffer(InputAssemblyDeviceReadbackPlanner::MaxSamplesPerFrame + 1, 1);
    const auto overflow = AmdGpu::NormalizeVertexInputRange(
        overflow_source, 0, 0, 16,
        Semantic(InputAssemblyDeviceReadbackPlanner::MaxSamplesPerFrame,
                 InputAssemblySourceKind::Vertex, 0));
    ASSERT_TRUE(overflow);
    EXPECT_FALSE(planner.Plan(*overflow).accepted);
    const auto plan = planner.EndFrame();
    EXPECT_EQ(plan.sample_count, InputAssemblyDeviceReadbackPlanner::MaxSamplesPerFrame);
    EXPECT_EQ(plan.semantic_count, InputAssemblyDeviceReadbackPlanner::MaxSamplesPerFrame);
    EXPECT_EQ(plan.loss.sample_capacity, 1);
    EXPECT_FALSE(plan.complete);
}

TEST(InputAssemblyDeviceReadbackPlan, RequiresCopyVisibilityAndRestoresInputReads) {
    const auto stream_vertex = AmdGpu::MakeInputAssemblyCopyBarrierPlan(
        InputAssemblyHostUsage::Stream, /*vertex=*/true, /*index=*/false);
    EXPECT_TRUE(stream_vertex.PreSourceHas(InputAssemblyCopyAccess::HostWrite));
    EXPECT_TRUE(stream_vertex.copy_reads_source);
    EXPECT_TRUE(stream_vertex.PostSourceHas(InputAssemblyCopyAccess::VertexRead));
    EXPECT_FALSE(stream_vertex.PostSourceHas(InputAssemblyCopyAccess::IndexRead));

    const auto device_both = AmdGpu::MakeInputAssemblyCopyBarrierPlan(
        InputAssemblyHostUsage::DeviceLocal, /*vertex=*/true, /*index=*/true);
    EXPECT_TRUE(device_both.PreSourceHas(InputAssemblyCopyAccess::TransferWrite));
    EXPECT_TRUE(device_both.PreSourceHas(InputAssemblyCopyAccess::ShaderWrite));
    EXPECT_TRUE(device_both.PostSourceHas(InputAssemblyCopyAccess::VertexRead));
    EXPECT_TRUE(device_both.PostSourceHas(InputAssemblyCopyAccess::IndexRead));
}

TEST(InputAssemblyDeviceReadbackPlan, InvalidatesMappedDestinationOnlyAfterCompletionOnce) {
    InputAssemblyReadbackCompletion completion{/*completion_tick=*/77};
    EXPECT_FALSE(completion.TryClaimInvalidation(76));
    EXPECT_TRUE(completion.TryClaimInvalidation(77));
    EXPECT_FALSE(completion.TryClaimInvalidation(78));
}

} // namespace
