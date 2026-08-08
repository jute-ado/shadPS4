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
using AmdGpu::InputAssemblyCaptureWindow;
using AmdGpu::InputAssemblyDeviceReadbackPlanner;
using AmdGpu::InputAssemblyHostUsage;
using AmdGpu::InputAssemblyImmediateReadbackReducer;
using AmdGpu::InputAssemblyImmediateSnapshot;
using AmdGpu::InputAssemblyReadbackCompletion;
using AmdGpu::InputAssemblyReadbackSlotPool;
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

TEST(InputAssemblyDeviceReadbackPlan, PostObtainStateDrivesBarrierAndAuthority) {
    const auto upload_cleared_gpu_state =
        AmdGpu::ResolveInputAssemblyPostObtainState(/*gpu_modified_after_obtain=*/false);
    EXPECT_FALSE(upload_cleared_gpu_state.needs_input_barrier);
    EXPECT_EQ(upload_cleared_gpu_state.authority, InputAssemblyAuthority::CpuAuthoritative);

    const auto gpu_resident_state =
        AmdGpu::ResolveInputAssemblyPostObtainState(/*gpu_modified_after_obtain=*/true);
    EXPECT_TRUE(gpu_resident_state.needs_input_barrier);
    EXPECT_EQ(gpu_resident_state.authority, InputAssemblyAuthority::GpuAuthoritative);
}

TEST(InputAssemblyDeviceReadbackPlan, UsesBoundedGenericFrameAndDrawWindows) {
    const auto defaults = InputAssemblyCaptureWindow::Defaults();
    EXPECT_TRUE(defaults.ContainsFrame(4000));
    EXPECT_TRUE(defaults.ContainsFrame(4999));
    EXPECT_FALSE(defaults.ContainsFrame(3999));
    EXPECT_FALSE(defaults.ContainsFrame(5000));
    EXPECT_TRUE(defaults.ContainsDraw(0));
    EXPECT_TRUE(defaults.ContainsDraw(1151));
    EXPECT_FALSE(defaults.ContainsDraw(1152));

    const InputAssemblyCaptureWindow band{/*frame_start=*/10, /*frame_count=*/3,
                                          /*draw_start=*/192, /*draw_count=*/192};
    EXPECT_TRUE(band.ContainsFrame(12));
    EXPECT_FALSE(band.ContainsFrame(13));
    EXPECT_TRUE(band.ContainsDraw(192));
    EXPECT_TRUE(band.ContainsDraw(383));
    EXPECT_FALSE(band.ContainsDraw(384));

    const InputAssemblyCaptureWindow overflow{/*frame_start=*/std::numeric_limits<u64>::max() - 1,
                                              /*frame_count=*/100,
                                              /*draw_start=*/std::numeric_limits<u32>::max() - 1,
                                              /*draw_count=*/100};
    EXPECT_TRUE(overflow.ContainsFrame(std::numeric_limits<u64>::max()));
    EXPECT_TRUE(overflow.ContainsDraw(std::numeric_limits<u32>::max()));
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
    for (u32 i = 0; i < vertex_decision.new_copy_count; ++i) {
        EXPECT_LT(vertex_decision.new_copy_indices[i], planner.CurrentFrame().sample_count);
    }
    for (u32 i = 0; i < index_decision.new_copy_count; ++i) {
        EXPECT_LT(index_decision.new_copy_indices[i], planner.CurrentFrame().sample_count);
    }
    EXPECT_LT(planner.EndFrame().sample_count,
              vertex_decision.reference_count + index_decision.reference_count);
    EXPECT_EQ(planner.EndFrame().semantic_count, 2);
}

TEST(InputAssemblyDeviceReadbackPlan, RepeatsPhysicalSamplesForEachCaptureDraw) {
    InputAssemblyDeviceReadbackPlanner planner;
    planner.BeginFrame(61);
    const auto first = AmdGpu::NormalizeVertexInputRange(
        Source(InputAssemblyHostUsage::DeviceLocal, 100, 512), 0, 0, 64,
        Semantic(5, InputAssemblySourceKind::Vertex, 0));
    const auto later = AmdGpu::NormalizeVertexInputRange(
        Source(InputAssemblyHostUsage::DeviceLocal, 100, 512), 0, 0, 64,
        Semantic(6, InputAssemblySourceKind::Vertex, 0));
    ASSERT_TRUE(first && later);

    const auto first_decision = planner.Plan(*first);
    const auto later_decision = planner.Plan(*later);
    ASSERT_TRUE(first_decision.accepted);
    ASSERT_TRUE(later_decision.accepted);
    EXPECT_EQ(first_decision.new_copy_count, 3);
    EXPECT_EQ(later_decision.new_copy_count, 3);

    const auto& plan = planner.EndFrame();
    ASSERT_EQ(plan.sample_count, 6);
    for (u32 i = 0; i < 3; ++i) {
        EXPECT_EQ(plan.samples[i].capture_draw, 5);
        EXPECT_EQ(plan.samples[i + 3].capture_draw, 6);
        EXPECT_EQ(plan.samples[i].source_offset, plan.samples[i + 3].source_offset);
    }
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
    EXPECT_EQ(InputAssemblyDeviceReadbackPlanner::MaxSamplesPerFrame, 8192);
    EXPECT_EQ(InputAssemblyDeviceReadbackPlanner::MaxSampleBytesPerFrame,
              8192 * InputAssemblyDeviceReadbackPlanner::MaxPhysicalSampleBytes);

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

TEST(InputAssemblyDeviceReadbackPlan, CpuConsumerOwnsFixedReadbackSlotUntilExplicitRelease) {
    InputAssemblyReadbackSlotPool pool;
    std::array<InputAssemblyReadbackSlotPool::Token, InputAssemblyReadbackSlotPool::MaxSlots>
        tokens{};
    for (auto& token : tokens) {
        const auto acquired = pool.TryAcquire();
        ASSERT_TRUE(acquired.has_value());
        token = *acquired;
    }
    EXPECT_FALSE(pool.TryAcquire().has_value());

    // GPU close/completion alone does not transfer CPU-consumer ownership back to the producer.
    EXPECT_FALSE(pool.TryAcquire().has_value());
    EXPECT_TRUE(pool.ReleaseAfterCpuConsume(tokens[3]));
    const auto recycled = pool.TryAcquire();
    ASSERT_TRUE(recycled.has_value());
    EXPECT_EQ(recycled->slot, tokens[3].slot);
    EXPECT_NE(recycled->generation, tokens[3].generation);
}

TEST(InputAssemblyDeviceReadbackPlan, RejectsDoubleStaleAndAbaSlotRelease) {
    InputAssemblyReadbackSlotPool pool;
    const auto first = pool.TryAcquire();
    ASSERT_TRUE(first.has_value());
    EXPECT_TRUE(pool.ReleaseAfterCpuConsume(*first));
    EXPECT_FALSE(pool.ReleaseAfterCpuConsume(*first));

    const auto replacement = pool.TryAcquire();
    ASSERT_TRUE(replacement.has_value());
    EXPECT_EQ(replacement->slot, first->slot);
    EXPECT_NE(replacement->generation, first->generation);
    EXPECT_FALSE(pool.ReleaseAfterCpuConsume(*first));
    EXPECT_TRUE(pool.ReleaseAfterCpuConsume(*replacement));

    EXPECT_FALSE(pool.ReleaseAfterCpuConsume({.slot = InputAssemblyReadbackSlotPool::MaxSlots,
                                             .generation = 1}));
    EXPECT_FALSE(pool.ReleaseAfterCpuConsume({}));
}

TEST(InputAssemblyDeviceReadbackPlan, ImmediateReducerComparesAcrossStreamReservations) {
    InputAssemblyImmediateReadbackReducer reducer;
    constexpr std::array a{std::byte{0x10}, std::byte{0x20}};
    constexpr std::array b{std::byte{0x10}, std::byte{0x30}};
    const auto semantic = Semantic(17, InputAssemblySourceKind::Vertex, 2);
    const auto observe = [&](u64 sequence, u32 reservation, u64 serial,
                             const auto& bytes) {
        return reducer.Observe(InputAssemblyImmediateSnapshot{
            .sequence = sequence,
            .semantic = semantic,
            .source = Buffer(std::numeric_limits<u32>::max(), reservation),
            .write_serial = serial,
            .authority = InputAssemblyAuthority::CpuAuthoritative,
            .bytes = std::span<const std::byte>{bytes},
            .complete = true,
        });
    };

    EXPECT_TRUE(observe(100, 1, 5, a).first_observation);
    const auto changed = observe(101, 2, 6, b);
    EXPECT_TRUE(changed.changed);
    EXPECT_TRUE(changed.source_changed);
    EXPECT_TRUE(changed.write_serial_changed);
    EXPECT_FALSE(changed.exact_aba_return);

    const auto returned = observe(102, 3, 7, a);
    EXPECT_TRUE(returned.changed);
    EXPECT_TRUE(returned.source_changed);
    EXPECT_TRUE(returned.write_serial_changed);
    EXPECT_TRUE(returned.exact_aba_return);
    EXPECT_FALSE(returned.stable_transport_aba);

    InputAssemblyImmediateReadbackReducer stable_reducer;
    const auto stable_observe = [&](u64 sequence, const auto& bytes) {
        return stable_reducer.Observe(InputAssemblyImmediateSnapshot{
            .sequence = sequence,
            .semantic = semantic,
            .source = Buffer(9, 4),
            .write_serial = 12,
            .authority = InputAssemblyAuthority::GpuAuthoritative,
            .bytes = std::span<const std::byte>{bytes},
            .complete = true,
        });
    };
    EXPECT_TRUE(stable_observe(200, a).first_observation);
    EXPECT_TRUE(stable_observe(201, b).changed);
    const auto stable_return = stable_observe(202, a);
    EXPECT_TRUE(stable_return.exact_aba_return);
    EXPECT_TRUE(stable_return.stable_transport_aba);
}

TEST(InputAssemblyDeviceReadbackPlan, ImmediateReducerSuppressesUnknownIncompleteAndGappedEvidence) {
    InputAssemblyImmediateReadbackReducer reducer;
    constexpr std::array bytes{std::byte{0x44}};
    const auto semantic = Semantic(9, InputAssemblySourceKind::Index, 0);
    const auto snapshot = [&](u64 sequence, InputAssemblyAuthority authority, bool complete) {
        return InputAssemblyImmediateSnapshot{
            .sequence = sequence,
            .semantic = semantic,
            .source = Buffer(4, 2),
            .write_serial = 8,
            .authority = authority,
            .bytes = std::span<const std::byte>{bytes},
            .complete = complete,
        };
    };

    EXPECT_TRUE(reducer.Observe(snapshot(10, InputAssemblyAuthority::Unknown, true))
                    .authority_ambiguous);
    EXPECT_TRUE(reducer.Observe(snapshot(11, InputAssemblyAuthority::GpuAuthoritative, false))
                    .incomplete);
    EXPECT_TRUE(reducer.Observe(snapshot(12, InputAssemblyAuthority::GpuAuthoritative, true))
                    .baseline_reset);
    EXPECT_TRUE(reducer.Observe(snapshot(14, InputAssemblyAuthority::GpuAuthoritative, true))
                    .sequence_gap);
}

TEST(InputAssemblyDeviceReadbackPlan, EmptyIncompleteSnapshotIsLossNotCapacityFailure) {
    InputAssemblyImmediateReadbackReducer reducer;
    const auto result = reducer.Observe(InputAssemblyImmediateSnapshot{
        .sequence = 44,
        .semantic = Semantic(3, InputAssemblySourceKind::Vertex, 1),
        .source = Buffer(8, 2),
        .write_serial = 5,
        .authority = InputAssemblyAuthority::GpuAuthoritative,
        .bytes = {},
        .complete = false,
    });
    EXPECT_TRUE(result.incomplete);
    EXPECT_FALSE(result.capacity_exceeded);
}

} // namespace
