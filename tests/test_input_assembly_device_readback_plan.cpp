// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>

#include "video_core/amdgpu/input_assembly_device_readback_plan.h"

namespace {

using AmdGpu::BoundInputAssemblySource;
using AmdGpu::InputAssemblyAuthority;
using AmdGpu::InputAssemblyBufferToken;
using AmdGpu::InputAssemblyCaptureWindow;
using AmdGpu::InputAssemblyCopyAccess;
using AmdGpu::InputAssemblyDeviceReadbackPlanner;
using AmdGpu::InputAssemblyHostUsage;
using AmdGpu::InputAssemblyImmediateChange;
using AmdGpu::InputAssemblyImmediateReadbackReducer;
using AmdGpu::InputAssemblyImmediateSnapshot;
using AmdGpu::InputAssemblyLagConfig;
using AmdGpu::InputAssemblyLagEventDetails;
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
    const auto first =
        AmdGpu::NormalizeVertexInputRange(Source(InputAssemblyHostUsage::DeviceLocal, 100, 512), 0,
                                          0, 64, Semantic(5, InputAssemblySourceKind::Vertex, 0));
    const auto later =
        AmdGpu::NormalizeVertexInputRange(Source(InputAssemblyHostUsage::DeviceLocal, 100, 512), 0,
                                          0, 64, Semantic(6, InputAssemblySourceKind::Vertex, 0));
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

    EXPECT_FALSE(pool.ReleaseAfterCpuConsume(
        {.slot = InputAssemblyReadbackSlotPool::MaxSlots, .generation = 1}));
    EXPECT_FALSE(pool.ReleaseAfterCpuConsume({}));
}

TEST(InputAssemblyDeviceReadbackPlan, ImmediateReducerComparesAcrossStreamReservations) {
    InputAssemblyImmediateReadbackReducer reducer;
    constexpr std::array a{std::byte{0x10}, std::byte{0x20}};
    constexpr std::array b{std::byte{0x10}, std::byte{0x30}};
    const auto semantic = Semantic(17, InputAssemblySourceKind::Vertex, 2);
    const auto observe = [&](u64 sequence, u32 reservation, u64 serial, const auto& bytes) {
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

TEST(InputAssemblyDeviceReadbackPlan,
     ImmediateReducerSuppressesUnknownIncompleteAndGappedEvidence) {
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

    EXPECT_TRUE(
        reducer.Observe(snapshot(10, InputAssemblyAuthority::Unknown, true)).authority_ambiguous);
    EXPECT_TRUE(
        reducer.Observe(snapshot(11, InputAssemblyAuthority::GpuAuthoritative, false)).incomplete);
    EXPECT_TRUE(reducer.Observe(snapshot(12, InputAssemblyAuthority::GpuAuthoritative, true))
                    .baseline_reset);
    EXPECT_TRUE(
        reducer.Observe(snapshot(14, InputAssemblyAuthority::GpuAuthoritative, true)).sequence_gap);
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

TEST(InputAssemblyDeviceReadbackPlan, LagReducerFindsExactAndJitteredHundredMillisecondReturns) {
    constexpr std::array a{std::byte{0x10}, std::byte{0x20}};
    constexpr std::array b{std::byte{0x10}, std::byte{0x30}};
    const auto semantic = Semantic(17, InputAssemblySourceKind::Vertex, 2);
    const auto snapshot = [&](u64 sequence, u64 process_time_us, const auto& bytes) {
        return InputAssemblyImmediateSnapshot{
            .sequence = sequence,
            .process_time_us = process_time_us,
            .semantic = semantic,
            .source = Buffer(9, 4),
            .write_serial = 12,
            .authority = InputAssemblyAuthority::GpuAuthoritative,
            .bytes = std::span<const std::byte>{bytes},
            .complete = true,
        };
    };

    InputAssemblyImmediateReadbackReducer exact{InputAssemblyLagConfig::Defaults()};
    EXPECT_TRUE(exact.Observe(snapshot(1, 0, a)).first_observation);
    EXPECT_FALSE(exact.Observe(snapshot(2, 100'000, b)).lag_exact_aba_return);
    const auto exact_return = exact.Observe(snapshot(3, 200'000, a));
    EXPECT_TRUE(exact_return.lag_exact_aba_return);
    EXPECT_TRUE(exact_return.lag_stable_transport_aba);
    EXPECT_FALSE(exact_return.lag_unavailable);

    InputAssemblyImmediateReadbackReducer jittered{InputAssemblyLagConfig{
        .enabled = true,
        .cadence_us = 100'000,
        .tolerance_us = 10'000,
    }};
    EXPECT_TRUE(jittered.Observe(snapshot(10, 0, a)).first_observation);
    EXPECT_FALSE(jittered.Observe(snapshot(11, 103'000, b)).lag_exact_aba_return);
    const auto jittered_return = jittered.Observe(snapshot(12, 207'000, a));
    EXPECT_TRUE(jittered_return.lag_exact_aba_return);
    EXPECT_TRUE(jittered_return.lag_stable_transport_aba);
}

TEST(InputAssemblyDeviceReadbackPlan, LagReducerReportsUnavailableOutsideStrictTolerance) {
    constexpr std::array a{std::byte{0x31}};
    constexpr std::array b{std::byte{0x42}};
    const auto semantic = Semantic(3, InputAssemblySourceKind::Index, 0);
    InputAssemblyImmediateReadbackReducer reducer{InputAssemblyLagConfig{
        .enabled = true,
        .cadence_us = 100'000,
        .tolerance_us = 25'000,
    }};
    const auto observe = [&](u64 sequence, u64 time, const auto& bytes) {
        return reducer.Observe({
            .sequence = sequence,
            .process_time_us = time,
            .semantic = semantic,
            .source = Buffer(2, 7),
            .write_serial = 5,
            .authority = InputAssemblyAuthority::GpuAuthoritative,
            .bytes = std::span<const std::byte>{bytes},
            .complete = true,
        });
    };

    observe(1, 0, a);
    observe(2, 60'000, b);
    const auto result = observe(3, 220'000, a);
    EXPECT_TRUE(result.lag_unavailable);
    EXPECT_FALSE(result.lag_exact_aba_return);
    EXPECT_FALSE(result.lag_stable_transport_aba);
}

TEST(InputAssemblyDeviceReadbackPlan, LagReducerRejectsSequenceAndTimeGaps) {
    constexpr std::array a{std::byte{0x51}};
    constexpr std::array b{std::byte{0x62}};
    const auto semantic = Semantic(7, InputAssemblySourceKind::Vertex, 1);
    const auto make = [&](u64 sequence, u64 time, const auto& bytes) {
        return InputAssemblyImmediateSnapshot{
            .sequence = sequence,
            .process_time_us = time,
            .semantic = semantic,
            .source = Buffer(4, 9),
            .write_serial = 3,
            .authority = InputAssemblyAuthority::GpuAuthoritative,
            .bytes = std::span<const std::byte>{bytes},
            .complete = true,
        };
    };

    InputAssemblyImmediateReadbackReducer sequence_gap{InputAssemblyLagConfig::Defaults()};
    sequence_gap.Observe(make(1, 0, a));
    EXPECT_TRUE(sequence_gap.Observe(make(3, 100'000, b)).sequence_gap);
    const auto after_sequence_gap = sequence_gap.Observe(make(4, 200'000, a));
    EXPECT_TRUE(after_sequence_gap.lag_unavailable);
    EXPECT_FALSE(after_sequence_gap.lag_exact_aba_return);

    InputAssemblyImmediateReadbackReducer time_gap{InputAssemblyLagConfig::Defaults()};
    time_gap.Observe(make(10, 100'000, a));
    EXPECT_TRUE(time_gap.Observe(make(11, 90'000, b)).time_gap);
    const auto after_time_gap = time_gap.Observe(make(12, 200'000, a));
    EXPECT_TRUE(after_time_gap.lag_unavailable);
    EXPECT_FALSE(after_time_gap.lag_exact_aba_return);
}

TEST(InputAssemblyDeviceReadbackPlan, LagContentAllowsTransportChangesButStableTransportDoesNot) {
    constexpr std::array a{std::byte{0x71}};
    constexpr std::array b{std::byte{0x82}};
    const auto semantic = Semantic(19, InputAssemblySourceKind::Vertex, 3);
    InputAssemblyImmediateReadbackReducer reducer{InputAssemblyLagConfig::Defaults()};
    const auto observe = [&](u64 sequence, u64 time, InputAssemblyBufferToken source, u64 serial,
                             InputAssemblyAuthority authority, const auto& bytes) {
        return reducer.Observe({
            .sequence = sequence,
            .process_time_us = time,
            .semantic = semantic,
            .source = source,
            .write_serial = serial,
            .authority = authority,
            .bytes = std::span<const std::byte>{bytes},
            .complete = true,
        });
    };

    observe(1, 0, Buffer(1, 1), 10, InputAssemblyAuthority::CpuAuthoritative, a);
    observe(2, 100'000, Buffer(1, 2), 11, InputAssemblyAuthority::CpuAuthoritative, b);
    const auto changed_transport =
        observe(3, 200'000, Buffer(1, 3), 12, InputAssemblyAuthority::CpuAuthoritative, a);
    EXPECT_TRUE(changed_transport.lag_exact_aba_return);
    EXPECT_FALSE(changed_transport.lag_stable_transport_aba);
    EXPECT_TRUE(changed_transport.lag_source_changed);
    EXPECT_TRUE(changed_transport.lag_write_serial_changed);

    const auto authority_transition =
        observe(4, 300'000, Buffer(1, 3), 12, InputAssemblyAuthority::GpuAuthoritative, b);
    EXPECT_TRUE(authority_transition.authority_ambiguous);
    EXPECT_TRUE(authority_transition.lag_ambiguous);
    EXPECT_FALSE(authority_transition.lag_exact_aba_return);
}

TEST(InputAssemblyDeviceReadbackPlan, LagReducerFindsMultiObservationEpisodeReturnsRepeatedly) {
    constexpr std::array a{std::byte{0x01}};
    constexpr std::array b{std::byte{0x02}};
    constexpr std::array c{std::byte{0x03}};
    const auto semantic = Semantic(29, InputAssemblySourceKind::Vertex, 4);
    InputAssemblyImmediateReadbackReducer reducer{InputAssemblyLagConfig::Defaults()};
    const auto observe = [&](u64 sequence, u64 time, const auto& bytes) {
        return reducer.Observe({
            .sequence = sequence,
            .process_time_us = time,
            .semantic = semantic,
            .source = Buffer(7, 2),
            .write_serial = 14,
            .authority = InputAssemblyAuthority::GpuAuthoritative,
            .bytes = std::span<const std::byte>{bytes},
            .complete = true,
        });
    };

    observe(1, 0, a);
    observe(2, 100'000, b);
    observe(3, 200'000, c);
    const auto first_return = observe(4, 300'000, a);
    EXPECT_TRUE(first_return.lag_episode_return);
    EXPECT_TRUE(first_return.lag_stable_transport_episode_return);
    EXPECT_EQ(first_return.lag_departure_sequence, 2);
    EXPECT_EQ(first_return.lag_departure_process_time_us, 100'000);
    EXPECT_EQ(first_return.lag_return_sequence, 4);
    EXPECT_EQ(first_return.lag_return_process_time_us, 300'000);

    observe(5, 400'000, b);
    const auto second_return = observe(6, 500'000, a);
    EXPECT_TRUE(second_return.lag_episode_return);
    EXPECT_EQ(second_return.lag_departure_sequence, 5);
    EXPECT_EQ(second_return.lag_return_sequence, 6);
}

TEST(InputAssemblyDeviceReadbackPlan, LagReducerLossBreaksEpisodeComparability) {
    constexpr std::array a{std::byte{0x21}};
    constexpr std::array b{std::byte{0x32}};
    const auto semantic = Semantic(31, InputAssemblySourceKind::Index, 0);
    InputAssemblyImmediateReadbackReducer reducer{InputAssemblyLagConfig::Defaults()};
    const auto make = [&](u64 sequence, u64 time, const auto& bytes, bool complete) {
        return InputAssemblyImmediateSnapshot{
            .sequence = sequence,
            .process_time_us = time,
            .semantic = semantic,
            .source = Buffer(8, 1),
            .write_serial = 6,
            .authority = InputAssemblyAuthority::GpuAuthoritative,
            .bytes = complete ? std::span<const std::byte>{bytes} : std::span<const std::byte>{},
            .complete = complete,
        };
    };

    reducer.Observe(make(1, 0, a, true));
    reducer.Observe(make(2, 100'000, b, true));
    EXPECT_TRUE(reducer.Observe(make(3, 200'000, b, false)).incomplete);
    const auto after_loss = reducer.Observe(make(4, 300'000, a, true));
    EXPECT_TRUE(after_loss.baseline_reset);
    EXPECT_TRUE(after_loss.lag_ambiguous);
    EXPECT_FALSE(after_loss.lag_episode_return);
    EXPECT_FALSE(after_loss.lag_exact_aba_return);
}

TEST(InputAssemblyDeviceReadbackPlan, LagEventDetailsAreBoundedAndExposeTimingOnly) {
    InputAssemblyLagEventDetails details;
    InputAssemblyImmediateChange change{
        .lag_episode_return = true,
        .lag_departure_sequence = 90,
        .lag_departure_process_time_us = 1'500'000,
        .lag_return_sequence = 96,
        .lag_return_process_time_us = 1'600'000,
    };
    for (u32 i = 0; i < InputAssemblyLagEventDetails::MaxEvents; ++i) {
        EXPECT_TRUE(details.Append(Semantic(i, InputAssemblySourceKind::Vertex, i), change));
    }
    EXPECT_FALSE(details.Append(Semantic(999, InputAssemblySourceKind::Index, 0), change));
    EXPECT_EQ(details.Events().size(), InputAssemblyLagEventDetails::MaxEvents);
    EXPECT_EQ(details.LostEvents(), 1);
    EXPECT_EQ(details.Events().front().departure_sequence, 90);
    EXPECT_EQ(details.Events().front().departure_process_time_us, 1'500'000);
    EXPECT_EQ(details.Events().front().return_sequence, 96);
    EXPECT_EQ(details.Events().front().return_process_time_us, 1'600'000);
}

TEST(InputAssemblyDeviceReadbackPlan, LagReducerBoundsHistoryAndReportsEvictedEvidence) {
    constexpr std::array a{std::byte{0x11}};
    constexpr std::array b{std::byte{0x22}};
    const auto semantic = Semantic(2, InputAssemblySourceKind::Vertex, 0);
    constexpr u64 cadence = InputAssemblyImmediateReadbackReducer::MaxLagHistoryObservations;
    InputAssemblyImmediateReadbackReducer reducer{InputAssemblyLagConfig{
        .enabled = true,
        .cadence_us = cadence,
        .tolerance_us = 0,
    }};

    InputAssemblyImmediateChange result{};
    for (u64 i = 0; i <= cadence * 2; ++i) {
        const auto& bytes = i == cadence ? b : a;
        result = reducer.Observe({
            .sequence = i + 1,
            .process_time_us = i,
            .semantic = semantic,
            .source = Buffer(3, 1),
            .write_serial = 4,
            .authority = InputAssemblyAuthority::GpuAuthoritative,
            .bytes = std::span<const std::byte>{bytes},
            .complete = true,
        });
    }
    EXPECT_LE(reducer.RetainedObservationCount(),
              InputAssemblyImmediateReadbackReducer::MaxLagHistoryObservations);
    EXPECT_TRUE(result.lag_history_loss);
    EXPECT_FALSE(result.lag_exact_aba_return);
}

TEST(InputAssemblyDeviceReadbackPlan, DisabledLagReducerRetainsNoState) {
    InputAssemblyImmediateReadbackReducer reducer{InputAssemblyLagConfig{
        .enabled = false,
        .cadence_us = 100'000,
        .tolerance_us = 25'000,
    }};
    constexpr std::array bytes{std::byte{0x7f}};
    const auto result = reducer.Observe({
        .sequence = 1,
        .process_time_us = 100,
        .semantic = Semantic(1, InputAssemblySourceKind::Vertex, 0),
        .source = Buffer(1, 1),
        .write_serial = 1,
        .authority = InputAssemblyAuthority::CpuAuthoritative,
        .bytes = std::span<const std::byte>{bytes},
        .complete = true,
    });
    EXPECT_TRUE(result.disabled);
    EXPECT_EQ(reducer.EntryCount(), 0);
    EXPECT_EQ(reducer.RetainedObservationCount(), 0);
}

} // namespace
