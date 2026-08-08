// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/amdgpu/input_assembly_device_integrity.h"

namespace {

using AmdGpu::InputAssemblyAuthority;
using AmdGpu::InputAssemblyBufferToken;
using AmdGpu::InputAssemblyDeviceIntegrityPlanner;
using AmdGpu::InputAssemblyDeviceIntegrityReducer;
using AmdGpu::InputAssemblyObservation;
using AmdGpu::InputAssemblyReadbackRing;
using AmdGpu::InputAssemblySemanticOrdinal;
using AmdGpu::InputAssemblySnapshot;
using AmdGpu::InputAssemblySourceKind;

constexpr InputAssemblyBufferToken Buffer(u32 slot, u32 generation) {
    return {.slot = slot, .generation = generation};
}

constexpr InputAssemblySemanticOrdinal Semantic(u32 draw, InputAssemblySourceKind kind,
                                                u32 binding) {
    return {.draw = draw, .kind = kind, .binding = binding};
}

constexpr InputAssemblyObservation Observation(
    InputAssemblySemanticOrdinal semantic, InputAssemblyBufferToken source, u64 offset, u64 size,
    u64 source_size, u64 write_serial = 1,
    InputAssemblyAuthority authority = InputAssemblyAuthority::GpuAuthoritative) {
    return {
        .semantic = semantic,
        .source = source,
        .source_offset = offset,
        .source_size = source_size,
        .size = size,
        .write_serial = write_serial,
        .authority = authority,
    };
}

template <std::size_t N>
InputAssemblySnapshot Snapshot(u64 sequence, u32 stable_identity, InputAssemblyBufferToken source,
                               u64 write_serial, InputAssemblyAuthority authority,
                               const std::array<std::byte, N>& bytes, bool complete = true) {
    return {
        .sequence = sequence,
        .stable_identity = stable_identity,
        .source = source,
        .write_serial = write_serial,
        .authority = authority,
        .bytes = std::span<const std::byte>{bytes},
        .complete = complete,
    };
}

TEST(InputAssemblyDeviceIntegrity, PreservesVertexAndIndexSemanticsWhileDeduplicatingOverlap) {
    InputAssemblyDeviceIntegrityPlanner planner;
    planner.BeginFrame(41);

    planner.Observe(
        Observation(Semantic(7, InputAssemblySourceKind::Vertex, 2), Buffer(11, 4), 100, 64, 512));
    planner.Observe(
        Observation(Semantic(7, InputAssemblySourceKind::Index, 0), Buffer(11, 4), 132, 64, 512));

    const auto plan = planner.EndFrame();
    ASSERT_TRUE(plan.complete);
    ASSERT_EQ(plan.sequence, 41);
    ASSERT_EQ(plan.range_count, 1);
    const auto& range = plan.ranges[0];
    EXPECT_EQ(range.source, Buffer(11, 4));
    EXPECT_EQ(range.source_offset, 100);
    EXPECT_EQ(range.size, 96);
    ASSERT_EQ(range.semantic_count, 2);
    EXPECT_EQ(range.semantics[0].semantic, Semantic(7, InputAssemblySourceKind::Vertex, 2));
    EXPECT_EQ(range.semantics[1].semantic, Semantic(7, InputAssemblySourceKind::Index, 0));
    EXPECT_NE(range.semantics[0].stable_identity, range.semantics[1].stable_identity);
}

TEST(InputAssemblyDeviceIntegrity, KeepsDistinctPhysicalBuffersSeparate) {
    InputAssemblyDeviceIntegrityPlanner planner;
    planner.BeginFrame(8);
    planner.Observe(
        Observation(Semantic(1, InputAssemblySourceKind::Vertex, 0), Buffer(3, 1), 0, 32, 64));
    planner.Observe(
        Observation(Semantic(1, InputAssemblySourceKind::Index, 0), Buffer(4, 1), 0, 32, 64));

    const auto plan = planner.EndFrame();
    EXPECT_TRUE(plan.complete);
    EXPECT_EQ(plan.range_count, 2);
}

TEST(InputAssemblyDeviceIntegrity, RejectsZeroOutOfBoundsAndOverflowingRanges) {
    InputAssemblyDeviceIntegrityPlanner planner;
    planner.BeginFrame(9);
    planner.Observe(
        Observation(Semantic(1, InputAssemblySourceKind::Vertex, 0), Buffer(1, 1), 0, 0, 64));
    planner.Observe(
        Observation(Semantic(1, InputAssemblySourceKind::Vertex, 1), Buffer(1, 1), 48, 17, 64));
    planner.Observe(Observation(Semantic(1, InputAssemblySourceKind::Index, 0), Buffer(1, 1),
                                std::numeric_limits<u64>::max() - 3, 8,
                                std::numeric_limits<u64>::max()));

    const auto plan = planner.EndFrame();
    EXPECT_FALSE(plan.complete);
    EXPECT_EQ(plan.range_count, 0);
    EXPECT_EQ(plan.loss.zero_size, 1);
    EXPECT_EQ(plan.loss.out_of_bounds, 2);
}

TEST(InputAssemblyDeviceIntegrity, DoesNotReportAuthorityChangesAsDeviceMismatch) {
    InputAssemblyDeviceIntegrityReducer reducer;
    constexpr std::array cpu_bytes{std::byte{0x10}, std::byte{0x20}};
    constexpr std::array gpu_bytes{std::byte{0x30}, std::byte{0x40}};

    const auto cpu = reducer.Observe(
        Snapshot(1, 17, Buffer(2, 9), 3, InputAssemblyAuthority::CpuAuthoritative, cpu_bytes));
    EXPECT_TRUE(cpu.authority_ambiguous);
    EXPECT_FALSE(cpu.changed);

    const auto gpu = reducer.Observe(
        Snapshot(2, 17, Buffer(2, 9), 3, InputAssemblyAuthority::GpuAuthoritative, gpu_bytes));
    EXPECT_TRUE(gpu.baseline_reset);
    EXPECT_FALSE(gpu.changed);
    EXPECT_FALSE(gpu.exact_aba_return);

    const auto unknown = reducer.Observe(
        Snapshot(3, 17, Buffer(2, 9), 3, InputAssemblyAuthority::Unknown, cpu_bytes));
    EXPECT_TRUE(unknown.authority_ambiguous);
    EXPECT_FALSE(unknown.changed);
    EXPECT_FALSE(unknown.exact_aba_return);
}

TEST(InputAssemblyDeviceIntegrity, MarksOverlappingWriteSerialsAmbiguous) {
    InputAssemblyDeviceIntegrityPlanner planner;
    planner.BeginFrame(10);
    planner.Observe(
        Observation(Semantic(3, InputAssemblySourceKind::Vertex, 0), Buffer(5, 2), 0, 32, 128, 6));
    planner.Observe(
        Observation(Semantic(4, InputAssemblySourceKind::Vertex, 0), Buffer(5, 2), 16, 32, 128, 7));

    const auto plan = planner.EndFrame();
    ASSERT_EQ(plan.range_count, 1);
    EXPECT_FALSE(plan.complete);
    EXPECT_TRUE(plan.ranges[0].write_serial_ambiguous);
    EXPECT_EQ(plan.loss.write_serial_ambiguity, 1);
}

TEST(InputAssemblyDeviceIntegrity, RejectsWriteSerialAndBufferGenerationChangesAsContentChanges) {
    InputAssemblyDeviceIntegrityReducer reducer;
    constexpr std::array a{std::byte{0x01}, std::byte{0x02}};
    constexpr std::array b{std::byte{0x03}, std::byte{0x04}};

    EXPECT_TRUE(
        reducer
            .Observe(Snapshot(1, 8, Buffer(4, 7), 10, InputAssemblyAuthority::GpuAuthoritative, a))
            .first_observation);

    const auto serial_change = reducer.Observe(
        Snapshot(2, 8, Buffer(4, 7), 11, InputAssemblyAuthority::GpuAuthoritative, b));
    EXPECT_TRUE(serial_change.write_serial_ambiguous);
    EXPECT_FALSE(serial_change.changed);

    const auto generation_change = reducer.Observe(
        Snapshot(3, 8, Buffer(4, 8), 11, InputAssemblyAuthority::GpuAuthoritative, a));
    EXPECT_TRUE(generation_change.source_generation_ambiguous);
    EXPECT_FALSE(generation_change.changed);
    EXPECT_FALSE(generation_change.exact_aba_return);

    const auto generation_aba = reducer.Observe(
        Snapshot(4, 8, Buffer(4, 7), 11, InputAssemblyAuthority::GpuAuthoritative, a));
    EXPECT_TRUE(generation_aba.source_generation_ambiguous);
    EXPECT_FALSE(generation_aba.changed);
    EXPECT_FALSE(generation_aba.exact_aba_return);
}

TEST(InputAssemblyDeviceIntegrity, EnforcesFixedRangeAndByteCapsWithExplicitLoss) {
    InputAssemblyDeviceIntegrityPlanner range_limited;
    range_limited.BeginFrame(11);
    for (u32 i = 0; i <= InputAssemblyDeviceIntegrityPlanner::MaxRangesPerFrame; ++i) {
        range_limited.Observe(Observation(Semantic(i, InputAssemblySourceKind::Vertex, 0),
                                          Buffer(i + 1, 1), 0, 1, 1));
    }
    const auto range_plan = range_limited.EndFrame();
    EXPECT_FALSE(range_plan.complete);
    EXPECT_EQ(range_plan.range_count, InputAssemblyDeviceIntegrityPlanner::MaxRangesPerFrame);
    EXPECT_EQ(range_plan.loss.range_capacity, 1);

    InputAssemblyDeviceIntegrityPlanner byte_limited;
    byte_limited.BeginFrame(12);
    byte_limited.Observe(Observation(Semantic(1, InputAssemblySourceKind::Index, 0), Buffer(1, 1),
                                     0, InputAssemblyDeviceIntegrityPlanner::MaxBytesPerFrame + 1,
                                     InputAssemblyDeviceIntegrityPlanner::MaxBytesPerFrame + 1));
    const auto byte_plan = byte_limited.EndFrame();
    EXPECT_FALSE(byte_plan.complete);
    EXPECT_EQ(byte_plan.range_count, 0);
    EXPECT_EQ(byte_plan.loss.byte_capacity, 1);
}

TEST(InputAssemblyDeviceIntegrity, EnforcesFixedSemanticHistoryCap) {
    InputAssemblyDeviceIntegrityPlanner planner;
    for (u32 i = 0; i <= InputAssemblyDeviceIntegrityPlanner::MaxHistoryIdentities; ++i) {
        planner.BeginFrame(i);
        planner.Observe(
            Observation(Semantic(i, InputAssemblySourceKind::Vertex, 0), Buffer(1, 1), 0, 1, 1));
        const auto plan = planner.EndFrame();
        if (i < InputAssemblyDeviceIntegrityPlanner::MaxHistoryIdentities) {
            EXPECT_TRUE(plan.complete) << i;
        } else {
            EXPECT_FALSE(plan.complete);
            EXPECT_EQ(plan.loss.history_capacity, 1);
        }
    }
}

TEST(InputAssemblyDeviceIntegrity, RingReservationsAreNonblockingAndAtomDisjoint) {
    InputAssemblyReadbackRing ring{/*capacity=*/128, /*non_coherent_atom_size=*/64};

    const auto first = ring.TryReserve(17, 1);
    const auto second = ring.TryReserve(1, 2);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(first->offset, 0);
    EXPECT_EQ(first->reserved_size, 64);
    EXPECT_EQ(second->offset, 64);
    EXPECT_EQ(second->reserved_size, 64);

    const auto busy = ring.TryReserve(1, 3);
    EXPECT_FALSE(busy.has_value());
    EXPECT_EQ(ring.BusyCount(), 1);

    ring.ReleaseCompleted(1);
    const auto wrapped = ring.TryReserve(33, 3);
    ASSERT_TRUE(wrapped.has_value());
    EXPECT_EQ(wrapped->offset, 0);
    EXPECT_EQ(wrapped->reserved_size, 64);
    EXPECT_FALSE(wrapped->Overlaps(*second));
}

TEST(InputAssemblyDeviceIntegrity, ReportsExactBytewiseChangesAndAbaOnlyWhenContiguousAndComplete) {
    InputAssemblyDeviceIntegrityReducer reducer;
    constexpr std::array a{std::byte{0x01}, std::byte{0x02}, std::byte{0x03}};
    constexpr std::array b{std::byte{0x01}, std::byte{0x09}, std::byte{0x03}};
    const auto token = Buffer(6, 3);

    EXPECT_TRUE(
        reducer.Observe(Snapshot(20, 22, token, 4, InputAssemblyAuthority::GpuAuthoritative, a))
            .first_observation);
    const auto changed =
        reducer.Observe(Snapshot(21, 22, token, 4, InputAssemblyAuthority::GpuAuthoritative, b));
    EXPECT_TRUE(changed.changed);
    EXPECT_FALSE(changed.exact_aba_return);
    const auto aba =
        reducer.Observe(Snapshot(22, 22, token, 4, InputAssemblyAuthority::GpuAuthoritative, a));
    EXPECT_TRUE(aba.changed);
    EXPECT_TRUE(aba.exact_aba_return);

    const auto gap =
        reducer.Observe(Snapshot(24, 22, token, 4, InputAssemblyAuthority::GpuAuthoritative, b));
    EXPECT_TRUE(gap.sequence_gap);
    EXPECT_FALSE(gap.changed);
    EXPECT_FALSE(gap.exact_aba_return);

    const auto incomplete = reducer.Observe(
        Snapshot(25, 22, token, 4, InputAssemblyAuthority::GpuAuthoritative, a, false));
    EXPECT_TRUE(incomplete.incomplete);
    EXPECT_FALSE(incomplete.changed);
    EXPECT_FALSE(incomplete.exact_aba_return);

    const auto after_incomplete =
        reducer.Observe(Snapshot(26, 22, token, 4, InputAssemblyAuthority::GpuAuthoritative, b));
    EXPECT_TRUE(after_incomplete.baseline_reset);
    EXPECT_FALSE(after_incomplete.exact_aba_return);
}

TEST(InputAssemblyDeviceIntegrity, DisabledGateInvokesNoCollectionOrAllocationCallback) {
    u32 collection_calls{};
    u32 allocation_calls{};
    const auto callback = [&] {
        ++collection_calls;
        ++allocation_calls;
        std::vector<std::byte> allocation(64);
        return !allocation.empty();
    };

    EXPECT_FALSE(AmdGpu::CollectInputAssemblyDeviceIntegrityIfEnabled(false, callback));
    EXPECT_EQ(collection_calls, 0);
    EXPECT_EQ(allocation_calls, 0);

    EXPECT_TRUE(AmdGpu::CollectInputAssemblyDeviceIntegrityIfEnabled(true, callback));
    EXPECT_EQ(collection_calls, 1);
    EXPECT_EQ(allocation_calls, 1);
}

} // namespace
