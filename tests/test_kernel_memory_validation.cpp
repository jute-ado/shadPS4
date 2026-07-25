// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>

#include "core/libraries/kernel/memory.h"
#include "core/memory.h"

namespace Libraries::Kernel {
namespace {

TEST(KernelMemoryValidation, RejectsMissingNamedDirectMemoryPointers) {
    void* address{};

    EXPECT_FALSE(IsMemoryAddressStorageValid(nullptr));
    EXPECT_FALSE(AreNamedMemoryPointersValid(nullptr, "anon"));
    EXPECT_FALSE(AreNamedMemoryPointersValid(&address, nullptr));
}

TEST(KernelMemoryValidation, AcceptsAddressStorageAndName) {
    void* address{};

    EXPECT_TRUE(IsMemoryAddressStorageValid(&address));
    EXPECT_TRUE(AreNamedMemoryPointersValid(&address, "anon"));
}

TEST(KernelMemoryValidation, ValidatesMemoryPoolReserveContract) {
    void* address{};

    EXPECT_TRUE(IsMemoryPoolReserveRequestValid(&address, 2_MB, 0));
    EXPECT_TRUE(IsMemoryPoolReserveRequestValid(&address, 2_MB, 2_MB));
    EXPECT_TRUE(IsMemoryPoolReserveRequestValid(&address, 4_MB, 4_MB));

    EXPECT_FALSE(IsMemoryPoolReserveRequestValid(nullptr, 2_MB, 0));
    EXPECT_FALSE(IsMemoryPoolReserveRequestValid(&address, 0, 0));
    EXPECT_FALSE(IsMemoryPoolReserveRequestValid(&address, 1_MB, 0));
    EXPECT_FALSE(IsMemoryPoolReserveRequestValid(&address, 2_MB, 1_MB));
    EXPECT_FALSE(IsMemoryPoolReserveRequestValid(&address, 2_MB, 6_MB));
}

TEST(KernelMemoryValidation, AlignmentMustMatchGranularityAndBeAPowerOfTwo) {
    EXPECT_TRUE(IsMemoryAlignmentValid(0, 16_KB));
    EXPECT_TRUE(IsMemoryAlignmentValid(16_KB, 16_KB));
    EXPECT_TRUE(IsMemoryAlignmentValid(32_KB, 16_KB));
    EXPECT_TRUE(IsMemoryAlignmentValid(4_MB, 2_MB));

    EXPECT_FALSE(IsMemoryAlignmentValid(0, 0));
    EXPECT_FALSE(IsMemoryAlignmentValid(8_KB, 16_KB));
    EXPECT_FALSE(IsMemoryAlignmentValid(48_KB, 16_KB));
    EXPECT_FALSE(IsMemoryAlignmentValid(1_MB, 2_MB));
    EXPECT_FALSE(IsMemoryAlignmentValid(6_MB, 2_MB));
}

TEST(KernelMemoryValidation, BatchRequestsRequireEntriesAndANonnegativeCount) {
    OrbisKernelBatchMapEntry entry{};

    EXPECT_TRUE(IsMemoryBatchRequestValid(&entry, 0));
    EXPECT_TRUE(IsMemoryBatchRequestValid(&entry, 1));
    EXPECT_FALSE(IsMemoryBatchRequestValid(nullptr, 0));
    EXPECT_FALSE(IsMemoryBatchRequestValid(&entry, -1));
}

TEST(KernelMemoryValidation, BatchMapOperationsStayWithinTheDefinedRange) {
    EXPECT_TRUE(IsMemoryBatchMapOperationValid(ORBIS_KERNEL_MAP_OP_MAP_DIRECT));
    EXPECT_TRUE(IsMemoryBatchMapOperationValid(ORBIS_KERNEL_MAP_OP_TYPE_PROTECT));

    EXPECT_FALSE(IsMemoryBatchMapOperationValid(-1));
    EXPECT_FALSE(IsMemoryBatchMapOperationValid(ORBIS_KERNEL_MAP_OP_TYPE_PROTECT + 1));
}

TEST(KernelMemoryValidation, PoolBatchRejectsUnsupportedOperations) {
    EXPECT_TRUE(IsMemoryPoolBatchOperationSupported(OrbisKernelMemoryPoolOpcode::Commit));
    EXPECT_TRUE(IsMemoryPoolBatchOperationSupported(OrbisKernelMemoryPoolOpcode::Decommit));
    EXPECT_TRUE(IsMemoryPoolBatchOperationSupported(OrbisKernelMemoryPoolOpcode::Protect));
    EXPECT_TRUE(IsMemoryPoolBatchOperationSupported(OrbisKernelMemoryPoolOpcode::TypeProtect));

    EXPECT_FALSE(IsMemoryPoolBatchOperationSupported(OrbisKernelMemoryPoolOpcode::Move));
    EXPECT_FALSE(
        IsMemoryPoolBatchOperationSupported(static_cast<OrbisKernelMemoryPoolOpcode>(0)));
}

TEST(KernelMemoryValidation, QueryRequestsRequireAnExactOutputContract) {
    OrbisVirtualQueryInfo info{};

    EXPECT_TRUE(IsMemoryQueryRequestValid(&info, sizeof(info), sizeof(info), 0));
    EXPECT_TRUE(IsMemoryQueryRequestValid(&info, sizeof(info), sizeof(info), 1));

    EXPECT_FALSE(IsMemoryQueryRequestValid(nullptr, sizeof(info), sizeof(info), 0));
    EXPECT_FALSE(IsMemoryQueryRequestValid(&info, sizeof(info) - 1, sizeof(info), 0));
    EXPECT_FALSE(IsMemoryQueryRequestValid(&info, sizeof(info) + 1, sizeof(info), 0));
    EXPECT_FALSE(IsMemoryQueryRequestValid(&info, sizeof(info), sizeof(info), -1));
    EXPECT_FALSE(IsMemoryQueryRequestValid(&info, sizeof(info), sizeof(info), 2));
}

TEST(KernelMemoryValidation, RequiredMemoryOutputsRejectNullStorage) {
    void* address{};
    u64 size{};

    EXPECT_TRUE(IsRequiredMemoryOutputValid(&address));
    EXPECT_TRUE(IsRequiredMemoryOutputValid(&size));
    EXPECT_FALSE(IsRequiredMemoryOutputValid(nullptr));
}

TEST(KernelMemoryValidation, RejectsWrappedVirtualMemoryContainmentRange) {
    const Core::VirtualMemoryArea area{
        .base = std::numeric_limits<VAddr>::max() - 0xff,
        .size = 0x80,
    };

    EXPECT_FALSE(area.Contains(std::numeric_limits<VAddr>::max() - 0x3f, 0x80));
}

TEST(KernelMemoryValidation, DetectsOverlapBeforeWrappedRangeEnd) {
    const Core::VirtualMemoryArea area{
        .base = std::numeric_limits<VAddr>::max() - 0x3f,
        .size = 0x30,
    };

    EXPECT_TRUE(area.Overlaps(std::numeric_limits<VAddr>::max() - 0x7f, 0x100));
}

TEST(KernelMemoryValidation, EmptyVirtualMemoryRangeDoesNotOverlap) {
    const Core::VirtualMemoryArea area{
        .base = 0x1000,
        .size = 0x1000,
    };

    EXPECT_FALSE(area.Overlaps(0x1800, 0));
}

TEST(KernelMemoryValidation, VirtualMemoryRangesUseHalfOpenEndpoints) {
    const Core::VirtualMemoryArea area{
        .base = 0x1000,
        .size = 0x1000,
    };

    EXPECT_TRUE(area.Contains(0x1000, 0x1000));
    EXPECT_TRUE(area.Contains(0x2000, 0));
    EXPECT_FALSE(area.Contains(0x2000, 1));
    EXPECT_FALSE(area.Overlaps(0x0800, 0x800));
    EXPECT_FALSE(area.Overlaps(0x2000, 0x800));
    EXPECT_TRUE(area.Overlaps(0x0800, 0x801));
    EXPECT_TRUE(area.Overlaps(0x1fff, 1));
}

TEST(KernelMemoryValidation, GpuRangeMayEndAtFortyBitBoundary) {
    constexpr VAddr gpu_address_limit = 0x10000000000;

    EXPECT_TRUE(Core::MemoryManager::IsValidGpuMapping(gpu_address_limit - 1, 1));
    EXPECT_TRUE(Core::MemoryManager::IsValidGpuMapping(0, gpu_address_limit));
    EXPECT_FALSE(Core::MemoryManager::IsValidGpuMapping(gpu_address_limit - 1, 2));
}

TEST(KernelMemoryValidation, RejectsWrappedGpuAddressRange) {
    EXPECT_FALSE(Core::MemoryManager::IsValidGpuMapping(
        std::numeric_limits<VAddr>::max() - 1, 4));
}

TEST(KernelMemoryValidation, FixedMappingsAlwaysRequireAValidVirtualRange) {
    using Core::MemoryMapFlags;

    EXPECT_TRUE(Core::IsRequestedMappingRangeValid(MemoryMapFlags::NoFlags, false));
    EXPECT_TRUE(Core::IsRequestedMappingRangeValid(MemoryMapFlags::Fixed, true));
    EXPECT_FALSE(Core::IsRequestedMappingRangeValid(MemoryMapFlags::Fixed, false));
    EXPECT_FALSE(Core::IsRequestedMappingRangeValid(
        MemoryMapFlags::Fixed | MemoryMapFlags::NoOverwrite, false));
}

TEST(KernelMemoryValidation, PoolDecommitValidatesEveryOverlappingArea) {
    const std::map<VAddr, Core::VirtualMemoryArea> areas{
        {0x1000,
         Core::VirtualMemoryArea{
             .base = 0x1000,
             .size = 0x1000,
             .type = Core::VMAType::PoolReserved,
         }},
        {0x2000,
         Core::VirtualMemoryArea{
             .base = 0x2000,
             .size = 0x1000,
             .type = Core::VMAType::Direct,
         }},
    };

    EXPECT_TRUE(Core::ArePoolDecommitAreasValid(areas.begin(), areas.end(), 0x1800, 0x800));
    EXPECT_FALSE(Core::ArePoolDecommitAreasValid(areas.begin(), areas.end(), 0x1800, 0x801));
    EXPECT_FALSE(Core::ArePoolDecommitAreasValid(std::next(areas.begin()), areas.end(), 0x2400,
                                                 0x100));
}

TEST(KernelMemoryValidation, PoolCommitAllowsAnExactBudgetFit) {
    EXPECT_TRUE(Core::CanCommitPoolBudget(0x1000, 0x1000));
    EXPECT_TRUE(Core::CanCommitPoolBudget(0x1001, 0x1000));
    EXPECT_FALSE(Core::CanCommitPoolBudget(0xfff, 0x1000));
}

TEST(KernelMemoryValidation, PoolBlockStatsKeepAvailableAndAllocatedCountsDistinct) {
    const auto counts = Core::ResolveMemoryPoolBlockCounts(3 * 64_KB, 2 * 64_KB);

    EXPECT_EQ(counts.available, 3);
    EXPECT_EQ(counts.allocated, 2);

    const auto partial_counts =
        Core::ResolveMemoryPoolBlockCounts(64_KB - 1, 2 * 64_KB - 1);
    EXPECT_EQ(partial_counts.available, 0);
    EXPECT_EQ(partial_counts.allocated, 1);
}

TEST(KernelMemoryValidation, PoolBackingReleaseIsAtomicAndReturnsAvailableBudget) {
    Core::PhysicalReleaseAccounting accounting{};

    accounting =
        Core::AccumulatePhysicalRelease(accounting, Core::PhysicalMemoryType::Pooled, 2 * 64_KB);
    EXPECT_TRUE(accounting.can_release);
    EXPECT_EQ(accounting.pool_bytes, 2 * 64_KB);

    accounting =
        Core::AccumulatePhysicalRelease(accounting, Core::PhysicalMemoryType::Allocated, 64_KB);
    EXPECT_TRUE(accounting.can_release);
    EXPECT_EQ(accounting.pool_bytes, 2 * 64_KB);

    accounting =
        Core::AccumulatePhysicalRelease(accounting, Core::PhysicalMemoryType::Committed, 64_KB);
    EXPECT_FALSE(accounting.can_release);
    EXPECT_EQ(accounting.pool_bytes, 2 * 64_KB);
}

TEST(KernelMemoryValidation, ValidatesRangesAndBudgetsWithoutOverflow) {
    EXPECT_TRUE(Core::IsMemoryRangeWithinLimit(0x1000, 0x400, 0x600));
    EXPECT_TRUE(Core::IsMemoryRangeWithinLimit(0x1000, 0x1000, 0));

    EXPECT_FALSE(Core::IsMemoryRangeWithinLimit(0x1000, 0x1001, 0));
    EXPECT_FALSE(Core::IsMemoryRangeWithinLimit(0x1000, 0xf00, 0x101));
    EXPECT_FALSE(Core::IsMemoryRangeWithinLimit(
        std::numeric_limits<u64>::max(), std::numeric_limits<u64>::max() - 1, 2));
}

TEST(KernelMemoryValidation, ValidatesBoundedMemoryWindowsWithoutOverflow) {
    EXPECT_TRUE(Core::IsMemoryRangeWithinBounds(0x1000, 0x4000, 0x1000, 0x3000));
    EXPECT_TRUE(Core::IsMemoryRangeWithinBounds(0x1000, 0x4000, 0x4000, 0));

    EXPECT_FALSE(Core::IsMemoryRangeWithinBounds(0x1000, 0x4000, 0xfff, 1));
    EXPECT_FALSE(Core::IsMemoryRangeWithinBounds(0x1000, 0x4000, 0x3000, 0x1001));
    EXPECT_FALSE(Core::IsMemoryRangeWithinBounds(0x4000, 0x1000, 0x2000, 0x100));
    EXPECT_FALSE(Core::IsMemoryRangeWithinBounds(
        0x1000, 0x4000, std::numeric_limits<u64>::max() - 0xfff, 0x2000));
}

TEST(KernelMemoryValidation, ResolvesAlignedMemoryRangesWithoutOverflow) {
    EXPECT_EQ(Core::ResolveAlignedMemoryRangeStart(0x11000, 0xc000, 0x4000, 0x40000), 0x18000u);
    EXPECT_EQ(Core::ResolveAlignedMemoryRangeStart(0x18000, 0xc000, 0x4000, 0x40000), 0x18000u);

    EXPECT_FALSE(Core::ResolveAlignedMemoryRangeStart(0x1000, 0, 0x1000, 0x40000).has_value());
    EXPECT_FALSE(Core::ResolveAlignedMemoryRangeStart(
                     std::numeric_limits<u64>::max() - 0x1000, 0x4000, 0x1000,
                     std::numeric_limits<u64>::max())
                     .has_value());
    EXPECT_FALSE(
        Core::ResolveAlignedMemoryRangeStart(0x3f000, 0x1000, 0x2000, 0x40000).has_value());
}

TEST(KernelMemoryValidation, AlignsMemoryValuesWithoutOverflow) {
    EXPECT_EQ(Core::AlignMemoryValueUp(0x4100, 0x4000), 0x8000u);
    EXPECT_EQ(Core::AlignMemoryValueUp(0x8000, 0x4000), 0x8000u);

    EXPECT_FALSE(Core::AlignMemoryValueUp(0x4000, 0).has_value());
    EXPECT_FALSE(Core::AlignMemoryValueUp(std::numeric_limits<u64>::max(), 0x4000).has_value());
}

TEST(KernelMemoryValidation, ResolvesFreeMemoryCandidatesInsideTheVma) {
    EXPECT_EQ(Core::ResolveFreeMemoryCandidate(0x1000, 0x3000, 0x1100, 0x1000, 0x1000),
              0x2000u);
    EXPECT_EQ(Core::ResolveFreeMemoryCandidate(0x1000, 0x3000, 0, 0x3000, 0x1000), 0x1000u);

    EXPECT_FALSE(
        Core::ResolveFreeMemoryCandidate(0x1000, 0x1000, 0x1800, 0x900, 0x1000).has_value());
    EXPECT_FALSE(Core::ResolveFreeMemoryCandidate(
                     std::numeric_limits<u64>::max() - 0xfff, 0xfff,
                     std::numeric_limits<u64>::max() - 0xff, 1, 0x1000)
                     .has_value());
    EXPECT_FALSE(
        Core::ResolveFreeMemoryCandidate(0x1000, 0x3000, 0x1000, 0x1000, 0).has_value());
}

TEST(KernelMemoryValidation, ResolvesPageAlignedProtectionSpansWithoutOverflow) {
    const auto span = Core::ResolvePageAlignedMemorySpan(0x4100, 0x4000, 0x4000);
    ASSERT_TRUE(span.has_value());
    EXPECT_EQ(span->base, 0x4000u);
    EXPECT_EQ(span->size, 0x8000u);

    const auto empty = Core::ResolvePageAlignedMemorySpan(0x4000, 0, 0x4000);
    ASSERT_TRUE(empty.has_value());
    EXPECT_EQ(empty->base, 0x4000u);
    EXPECT_EQ(empty->size, 0u);

    EXPECT_FALSE(Core::ResolvePageAlignedMemorySpan(0x4000, 0x1000, 0).has_value());
    EXPECT_FALSE(Core::ResolvePageAlignedMemorySpan(
                     1, std::numeric_limits<u64>::max(), 0x4000)
                     .has_value());
    EXPECT_FALSE(Core::ResolvePageAlignedMemorySpan(
                     std::numeric_limits<u64>::max() - 0xff, 0x100, 0x4000)
                     .has_value());
}

TEST(KernelMemoryValidation, ResolvesBoundedAvailableMemorySpansAfterAlignment) {
    const auto aligned =
        Core::ResolveAvailableMemorySpan(0x1000, 0x9000, 0x2500, 0x8000, 0x2000);
    ASSERT_TRUE(aligned.has_value());
    EXPECT_EQ(aligned->base, 0x4000u);
    EXPECT_EQ(aligned->size, 0x4000u);

    const auto unaligned =
        Core::ResolveAvailableMemorySpan(0x1000, 0x9000, 0x2500, 0x8000, 0);
    ASSERT_TRUE(unaligned.has_value());
    EXPECT_EQ(unaligned->base, 0x2500u);
    EXPECT_EQ(unaligned->size, 0x5b00u);

    EXPECT_FALSE(
        Core::ResolveAvailableMemorySpan(0x1000, 0x1000, 0x3000, 0x4000, 0x1000).has_value());
    EXPECT_FALSE(Core::ResolveAvailableMemorySpan(
                     std::numeric_limits<u64>::max() - 0x1000, 0x2000, 0,
                     std::numeric_limits<u64>::max(), 0x1000)
                     .has_value());
    EXPECT_FALSE(Core::ResolveAvailableMemorySpan(
                     std::numeric_limits<u64>::max() - 0x1000, 0x1000, 0,
                     std::numeric_limits<u64>::max(), 0x4000)
                     .has_value());
}

TEST(KernelMemoryValidation, ClipsAndIntersectsDirectMemoryReleaseSpans) {
    const auto clipped = Core::ClipMemorySpanToLimit(0x3000, 0x3000, 0x4000);
    ASSERT_TRUE(clipped.has_value());
    EXPECT_EQ(clipped->base, 0x3000u);
    EXPECT_EQ(clipped->size, 0x1000u);

    const auto wrapped = Core::ClipMemorySpanToLimit(
        std::numeric_limits<u64>::max() - 0x1000, 0x4000,
        std::numeric_limits<u64>::max());
    ASSERT_TRUE(wrapped.has_value());
    EXPECT_EQ(wrapped->base, std::numeric_limits<u64>::max() - 0x1000);
    EXPECT_EQ(wrapped->size, 0x1000u);

    const auto allocated_overlap =
        Core::IntersectMemorySpans({.base = 0x1000, .size = 0x3000},
                                   {.base = 0x2800, .size = 0x2000});
    ASSERT_TRUE(allocated_overlap.has_value());
    EXPECT_EQ(allocated_overlap->base, 0x2800u);
    EXPECT_EQ(allocated_overlap->size, 0x1800u);

    const auto mapping_overlap =
        Core::IntersectMemorySpans({.base = 0x1000, .size = 0x1000},
                                   {.base = 0x1800, .size = 0x1000});
    ASSERT_TRUE(mapping_overlap.has_value());
    EXPECT_EQ(mapping_overlap->base, 0x1800u);
    EXPECT_EQ(mapping_overlap->size, 0x800u);

    EXPECT_FALSE(Core::ClipMemorySpanToLimit(0x4000, 0x1000, 0x4000).has_value());
    EXPECT_FALSE(Core::IntersectMemorySpans({.base = 0x1000, .size = 0x1000},
                                            {.base = 0x2000, .size = 0x1000})
                     .has_value());
}

TEST(KernelMemoryValidation, RejectsWrappedPhysicalAreaAdjacency) {
    Core::PhysicalMemoryArea area{
        .base = std::numeric_limits<PAddr>::max() - 0xff,
        .size = 0x200,
    };
    Core::PhysicalMemoryArea low_area{
        .base = 0x100,
        .size = 0x100,
    };

    EXPECT_FALSE(area.CanMergeWith(low_area));
}

TEST(KernelMemoryValidation, RejectsWrappedVirtualAreaAdjacency) {
    Core::VirtualMemoryArea area{
        .base = std::numeric_limits<VAddr>::max() - 0xff,
        .size = 0x200,
    };
    Core::VirtualMemoryArea low_area{
        .base = 0x100,
        .size = 0x100,
    };

    EXPECT_FALSE(area.CanMergeWith(low_area));
}

TEST(KernelMemoryValidation, AdjacentMemoryAreasRemainMergeable) {
    Core::PhysicalMemoryArea physical{
        .base = 0x1000,
        .size = 0x1000,
    };
    Core::PhysicalMemoryArea next_physical{
        .base = 0x2000,
        .size = 0x1000,
    };
    Core::VirtualMemoryArea virtual_area{
        .base = 0x1000,
        .size = 0x1000,
    };
    Core::VirtualMemoryArea next_virtual{
        .base = 0x2000,
        .size = 0x1000,
    };

    EXPECT_TRUE(physical.CanMergeWith(next_physical));
    EXPECT_TRUE(virtual_area.CanMergeWith(next_virtual));
}

} // namespace
} // namespace Libraries::Kernel
